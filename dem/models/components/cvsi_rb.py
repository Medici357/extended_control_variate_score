"""Rao-Blackwellised / rotation-sampled CVSI for the iDEM SNIS estimator.
VE schedule (a(t)=1), non-molecular energies only.

Estimator:  s_hat = sum_k w_k f_k,   w = softmax(log p(x0_k) + log q_k)
  q_k = quadrature weight of node k (constant for iid draws and for rotation nodes). The numerator and the
  denominator of the SNIS ratio are both integrated, never the score alone.

Proposal-noise structures (nodes are grouped: K = n_A groups x n_q nodes, node index k = a * n_q + j)
  none    iid N(0, I) draws (identical to repo CVSI)
  coord   conditional MC along a random line, nodes at stratified / Gauss-Hermite quantiles of the line coordinate
  radial  D = 2 only: random radius, n_q equally spaced angles with a random phase
  rot     ROTATION SAMPLING (Glynn & Szechtman) in any D:  z_j = R^j z_0,  z_0 ~ N(0, I),  R = block rotation by
          2*pi*k_i/n_q in disjoint coordinate planes (k_i coprime to n_q; a leftover odd coordinate is reflected).
          Every node is exactly N(0, I) (an orthogonal map of an isotropic Gaussian), and for even n_q the group sum
          of the nodes is exactly 0.   n_q = 2  ->  antithetic pairs (z, -z).   In D = 2 this is the same
          distribution as `radial`.
          rb_rot_radius = m > 0 additionally rotates the radius quantile by m*j/n_q (mod 1) in every plane
          (a randomly shifted rank-1 lattice, i.e. Cranley-Patterson rotation); marginals stay exactly N(0, I).

Control-variate coefficient (coef_weighting)
  prop   proposal-weighted node-level OLS (the repo's unweighted statistics)
  post   posterior(SNIS)-weighted node-level OLS
  gpost  posterior-weighted GROUP-level OLS: nodes inside a group are dependent, groups are independent, so the
         coefficient is fit on per-group influence sums (G_a = sum_j w (g - s0), P_a = sum_j w p).
         (An UNWEIGHTED group-level fit is degenerate for rotation noise: the group sum of d = z / sigma is
         exactly 0, so the fit would return c = 1.)  Used only where RB is active and the group ESS >= min_ess;
         elsewhere falls back to `prop`.

Earlier fixes kept: float32-safe inverse CDF (eps clamp), repo-identical single-control coefficient (no denominator
clamp, non-finite c -> 1.0).
"""
import math

import numpy as np
import torch

from .cvsi import _LOG_W_FLOOR, clip_scores, get_a2_b2


def _std_normal_icdf(u):
    eps = torch.finfo(u.dtype).eps
    u = u.clamp(min=eps, max=1.0 - eps)  # keeps 2u-1 strictly inside (-1, 1) in this dtype
    return math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0)


def _gauss_hermite(n, device, dtype):
    x, w = np.polynomial.hermite_e.hermegauss(n)  # weight exp(-x^2/2)
    w = w / w.sum()  # -> probabilities
    return torch.as_tensor(x, device=device, dtype=dtype), torch.as_tensor(w, device=device, dtype=dtype)


def _random_frames(B, D, device, dtype):
    q, r = torch.linalg.qr(torch.randn(B, D, D, device=device, dtype=dtype))
    return q * torch.sign(torch.diagonal(r, dim1=-2, dim2=-1)).unsqueeze(-2)


def _coprime_mults(n):
    ks = [k for k in range(1, n) if math.gcd(k, n) == 1]
    return ks if ks else [1]


def _rotation_nodes(n_A, n_q, B, D, rot_radius, device, dtype):
    """z_j = R^j z_0 (optionally with a lattice-rotated radius).  Returns (n_A, n_q, B, D)."""
    assert n_q >= 2, "rotation sampling needs rb_nq >= 2 (rb_nq = 2 is antithetic pairs)"
    z0 = torch.randn(n_A, 1, B, D, device=device, dtype=dtype)
    j = torch.arange(n_q, device=device, dtype=dtype)
    P = D // 2
    parts = []
    if P > 0:
        ks = _coprime_mults(n_q)
        k = torch.tensor([ks[i % len(ks)] for i in range(P)], device=device, dtype=dtype)  # (P,)
        a, b = z0[..., 0 : 2 * P : 2], z0[..., 1 : 2 * P : 2]  # (n_A, 1, B, P)
        ang = 2.0 * math.pi * j.view(1, n_q, 1, 1) * k.view(1, 1, 1, P) / n_q  # (1, n_q, 1, P)
        if rot_radius == 0:
            c, s = torch.cos(ang), torch.sin(ang)
            oa, ob = c * a - s * b, s * a + c * b
        else:
            u0 = torch.exp(-0.5 * (a * a + b * b))  # radius quantile: Uniform(0, 1)
            shift = (float(rot_radius) * j / n_q).view(1, n_q, 1, 1)
            u = torch.remainder(u0 + shift, 1.0).clamp_min(torch.finfo(dtype).tiny)
            r = torch.sqrt(-2.0 * torch.log(u))
            th = torch.atan2(b, a) + ang
            oa, ob = r * torch.cos(th), r * torch.sin(th)
        parts.append(torch.stack([oa, ob], dim=-1).reshape(n_A, n_q, B, 2 * P))
    if D % 2 == 1:  # leftover coordinate cannot rotate: reflect it (marginal stays N(0, 1))
        sign = (1.0 - 2.0 * torch.remainder(j, 2.0)).view(1, n_q, 1, 1)
        parts.append(z0[..., -1:] * sign)
    return torch.cat(parts, dim=-1)


def rb_noise(K, B, D, mode, rule, n_q, random_frame, device, dtype, rot_radius=0):
    """Structured proposal noise.  Returns z (K,B,D) and log_q (K,B).
    K = n_A * n_q: n_A random lines / circles / rotation orbits, n_q nodes on each. Node index k = a * n_q + j."""
    assert K % n_q == 0, f"K={K} must be divisible by rb_nq={n_q}"
    n_A = K // n_q
    if mode == "coord":
        assert D >= 2, "coord mode needs D >= 2"
        a = torch.randn(n_A, 1, B, D - 1, device=device, dtype=dtype).expand(n_A, n_q, B, D - 1)
        if rule == "gh":
            nu, q = _gauss_hermite(n_q, device, dtype)
            nu = nu.view(1, n_q, 1, 1).expand(n_A, n_q, B, 1)
            log_q = q.log().view(1, n_q, 1).expand(n_A, n_q, B)
        elif rule == "strat":
            j = torch.arange(n_q, device=device, dtype=dtype).view(1, n_q, 1)
            u = (j + torch.rand(n_A, n_q, B, device=device, dtype=dtype)) / n_q
            nu = _std_normal_icdf(u).unsqueeze(-1)  # clamped inside the icdf
            log_q = torch.full((n_A, n_q, B), -math.log(n_q), device=device, dtype=dtype)
        else:
            raise ValueError(f"coord mode needs rb_rule in (gh, strat), got {rule!r}")
        z = torch.cat([a, nu], dim=-1)  # (n_A, n_q, B, D)
        if random_frame:  # random axes: no preferred direction
            Q = _random_frames(B, D, device, dtype)
            z = torch.einsum("bde,anbe->anbd", Q, z)
    elif mode == "radial":
        assert D == 2 and rule == "trap", "radial mode is implemented for D == 2 with rb_rule='trap'"
        r = torch.sqrt(-2.0 * torch.log(torch.rand(n_A, 1, B, device=device, dtype=dtype).clamp_min(1e-12)))
        phi = 2 * math.pi * torch.rand(n_A, 1, B, device=device, dtype=dtype)  # random phase -> unbiased
        th = phi + 2 * math.pi * torch.arange(n_q, device=device, dtype=dtype).view(1, n_q, 1) / n_q
        z = torch.stack([r * torch.cos(th), r * torch.sin(th)], dim=-1)  # (n_A, n_q, B, 2)
        log_q = torch.zeros(n_A, n_q, B, device=device, dtype=dtype)
    elif mode == "rot":
        z = _rotation_nodes(n_A, n_q, B, D, int(rot_radius), device, dtype)  # (n_A, n_q, B, D)
        log_q = torch.zeros(n_A, n_q, B, device=device, dtype=dtype)
    else:
        raise ValueError(f"unknown rb_mode {mode!r}")
    return z.reshape(K, B, D), log_q.reshape(K, B)


def _single_control_coef(g, d, m):
    """Weighted OLS coefficient of CVSI's one control p = g - d. m: (K,B) weights summing to 1.
    With m = 1/K this is exactly (Vg - Cgd) / (Vg + Vd - 2 Cgd), the repo's c-tilde (a = 1)."""
    p = g - d
    mm = m.unsqueeze(-1)
    pc = p - (mm * p).sum(0, keepdim=True)
    gc = g - (mm * g).sum(0, keepdim=True)
    num = (mm * pc * gc).sum(dim=(0, 2))
    den = (mm * pc * pc).sum(dim=(0, 2))
    return num / den  # (B,); non-finite handled by the caller (repo fallback c -> 1)


def _group_post_coef(g, p, w, n_q):
    """Group-level (cluster-robust) posterior-weighted OLS for the SNIS ratio.
    g, p: (K,B,D) unclipped integrands (p = g - d);  w: (K,B) normalised SNIS weights.
    Delta-method influence terms per group a:  G_a = sum_j w (g - s0),  P_a = sum_j w p   (s0 = SNIS mean of g).
    Groups are independent, nodes inside a group are not, so c is fit across groups (trace over D)."""
    K, B, D = g.shape
    A = K // n_q
    wk = w.unsqueeze(-1)
    s0 = (wk * g).sum(0, keepdim=True)
    G = (wk * (g - s0)).reshape(A, n_q, B, D).sum(1)  # (A,B,D)
    P = (wk * p).reshape(A, n_q, B, D).sum(1)
    Gc = G - G.mean(0, keepdim=True)
    Pc = P - P.mean(0, keepdim=True)
    return (Gc * Pc).sum(dim=(0, 2)) / (Pc * Pc).sum(dim=(0, 2))  # (B,)


def idem_score_cvsi_rb_fn(
    t, x, neg_energy_function, noise_sch, num_mc_samples,
    debug=False, pointwise=False,
    divide_by_dim=False, weighted_stats=False, ct_per_molecule=False,    # accepted and ignored (drop-in signature)
    clipper_tsi=None,
    rb_mode="coord",              # none | coord | radial | rot
    rb_rule="strat",              # strat | gh (coord)   trap (radial)   ignored for rot
    rb_nq=10,                     # nodes per group; groups = K / rb_nq   (rot: rb_nq = 2 is antithetic)
    rb_random_frame=True,
    rb_max_sigma=None,            # use RB only where sqrt(h_t) <= this; iid draws elsewhere (same K)
    rb_rot_radius=0,              # rot only: radius-quantile lattice multiplier m (0 = pure rotation)
    coef_weighting="prop",        # prop | post | gpost
    min_ess=8.0,
):
    if getattr(neg_energy_function, "is_molecule", False):
        raise NotImplementedError("RB-CVSI is implemented for non-molecular energies only")
    with torch.no_grad():
        if t.ndim == 0:
            t = t.unsqueeze(0).repeat(len(x))
        K, (B, D) = num_mc_samples, x.shape
        a2, b2 = get_a2_b2(t, noise_sch)                                  # (B,)
        assert torch.allclose(a2, torch.ones_like(a2)), "VE schedule only"
        h_t = b2.view(1, B, 1)

        z = torch.randn(K, B, D, device=x.device, dtype=x.dtype)          # plain iid draws (identical to repo CVSI)
        log_q = torch.zeros(K, B, device=x.device, dtype=x.dtype)
        use = torch.zeros(B, dtype=torch.bool, device=x.device)           # points where structured noise is active
        if rb_mode != "none":
            z_rb, lq_rb = rb_noise(K, B, D, rb_mode, rb_rule, rb_nq, rb_random_frame, x.device, x.dtype,
                                   rot_radius=rb_rot_radius)
            use = torch.ones(B, dtype=torch.bool, device=x.device) if rb_max_sigma is None else (b2.sqrt() <= rb_max_sigma)
            z = torch.where(use[None, :, None], z_rb, z)
            log_q = torch.where(use[None, :], lq_rb, log_q)

        x0 = x.unsqueeze(0) + z * h_t.sqrt()
        log_p = torch.nan_to_num(neg_energy_function(x0).reshape(K, B),
                                 nan=_LOG_W_FLOOR, posinf=_LOG_W_FLOOR, neginf=_LOG_W_FLOOR)
        w = torch.softmax(log_p + log_q, dim=0).unsqueeze(-1)             # (K,B,1)

        grad_fn = torch.func.grad(lambda v: neg_energy_function(v).sum())
        g_u = torch.nan_to_num(grad_fn(x0), nan=0.0, posinf=0.0, neginf=0.0)     # unclipped: statistics
        g_c = g_u if clipper_tsi is None else clip_scores(g_u, clipper_tsi, neg_energy_function)
        d = (x0 - x.unsqueeze(0)) / h_t                                    # DSI integrand

        c = _single_control_coef(g_u, d, torch.softmax(log_q, dim=0))      # proposal-weighted statistics
        if coef_weighting == "post":
            ess = 1.0 / (w.squeeze(-1) ** 2).sum(0)
            c = torch.where(ess >= min_ess, _single_control_coef(g_u, d, w.squeeze(-1)), c)
        elif coef_weighting == "gpost":
            if rb_mode != "none":
                wk = w.squeeze(-1)
                c_g = _group_post_coef(g_u, g_u - d, wk, rb_nq)
                ess_g = 1.0 / (wk.reshape(K // rb_nq, rb_nq, B).sum(1) ** 2).sum(0)   # ESS counted in groups
                ok = use & (ess_g >= min_ess) & torch.isfinite(c_g)
                c = torch.where(ok, c_g, c)
        elif coef_weighting != "prop":
            raise ValueError(coef_weighting)
        c = torch.where(torch.isfinite(c), c, torch.ones_like(c))          # repo fallback: c -> 1 (pure DSI)

        tilde_ct = c.view(1, B, 1)
        f = g_c - tilde_ct * (g_c - d)                                     # == (1-c) g_c + c d, as in the repo
        score = (w * f).sum(dim=0)
        if debug:
            return score, w, tilde_ct, g_c, d, f, log_p.unsqueeze(-1)      # same 7-tuple as idem_score_cvsi_fn
        if pointwise:
            return w * f, w, tilde_ct
        return score