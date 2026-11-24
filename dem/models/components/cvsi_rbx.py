"""CVSI estimator with: stratified rotation sampling + defensive mixture + multi-control (all optional).
VE schedule only, non-molecular energies only.  Drop-in replacement for idem_score_cvsi_rb_fn.

Target at a noisy point x_t (noise std sigma):  pi(x) = p(x) * N(x; x_t, sigma^2 I).   Score = E_pi[grad log p].
Estimator:  s_hat = sum_k w_k f_k,   w = softmax( log p(x_k) + log_ratio_k ),   log_ratio_k = log N(x_k;x_t,sigma^2) - log q_mix(x_k)

1) rb_mode = "rotstrat"  (stratification + Rao-Blackwellisation, 2D)
     n_A groups of n_q nodes.  Each group is one circle: radius r, random phase, n_q equally spaced angles.
     RADIUS is STRATIFIED over the n_A groups (u = (a + U)/n_A, r = sqrt(-2 ln(1-u))) -> radii cover [0,inf) evenly.
     ANGLE is Rao-Blackwellised by the rotation (n_q angles averaged exactly).  Every node is exactly N(0, I).
   rb_mode = "rot"   plain rotation sampling (random radius per group);  "none" = i.i.d. draws (repo CVSI).

2) Defensive mixture (def_alpha > 0, only where sqrt(h_t) >= def_min_sigma, i.e. high noise)
     Proposal  q_mix = (1-a) N(x_t, sigma^2 I) + a N(c, s^2 I),   a = n_def/n_A  (a fixed number of groups, "deterministic mixture").
     The second, broad component is centred on the data region (c = def_center, s = def_scale, in normalised units).
     Weights use the mixture density (balance heuristic), so the estimator stays unbiased and no weight can exceed p/(a q_def).
     The defensive groups use the same stratified-rotation sampler, rotating about c.

3) Multi-control (multi_control=True): jointly fit  [h, logp-control, linear-x, linear-y]  with squared-SNIS-weight least squares
     (shared coefficient over the output dims).  Each control is  phi * (g - d) + grad(phi)  and has mean zero under pi (Stein identity).
     Used only at points whose ESS >= mc_min_ess (few effective draws cannot pay for several coefficients); other points use the
     usual one-control CVSI coefficient.
"""
import math

import torch

from .cvsi import _LOG_W_FLOOR, clip_scores, get_a2_b2
from .cvsi_rb import _rotation_nodes, _single_control_coef


def _polar_strat_rot(n, n_q, B, device, dtype):
    """(n, n_q, B, 2) standard-normal nodes: radius stratified over the n groups, angle = random phase + equal steps."""
    a = torch.arange(n, device=device, dtype=dtype).view(n, 1, 1)
    u = (a + torch.rand(n, 1, B, device=device, dtype=dtype)) / n  # stratified uniform
    surv = (1.0 - u).clamp_min(1e-12)
    r = torch.sqrt(-2.0 * torch.log(surv))  # Rayleigh radius of a 2D standard normal
    phi = 2.0 * math.pi * torch.rand(n, 1, B, device=device, dtype=dtype)
    j = torch.arange(n_q, device=device, dtype=dtype).view(1, n_q, 1)
    th = phi + 2.0 * math.pi * j / n_q
    return torch.stack([r * torch.cos(th), r * torch.sin(th)], dim=-1)


def _block(mode, n, n_q, B, D, device, dtype):
    """(n, n_q, B, D) nodes, each exactly N(0, I)."""
    if n == 0:
        return torch.zeros(0, n_q, B, D, device=device, dtype=dtype)
    if mode == "rotstrat":
        assert D == 2, "rotstrat is implemented for D == 2"
        return _polar_strat_rot(n, n_q, B, device, dtype)
    if mode == "rot":
        return _rotation_nodes(n, n_q, B, D, 0, device, dtype)
    if mode == "none":
        return torch.randn(n, n_q, B, D, device=device, dtype=dtype)
    raise ValueError(f"rb_mode must be none|rot|rotstrat, got {mode!r}")


def _log_gauss(x, mean, var, D):
    var = torch.as_tensor(var, device=x.device, dtype=x.dtype)
    return -0.5 * D * torch.log(2.0 * math.pi * var) - ((x - mean) ** 2).sum(-1) / (2.0 * var)


def idem_score_cvsi_rbx_fn(
    t, x, neg_energy_function, noise_sch, num_mc_samples,
    debug=False, pointwise=False,
    divide_by_dim=False, weighted_stats=False, ct_per_molecule=False,    # accepted and ignored (drop-in signature)
    clipper_tsi=None,
    rb_mode="rotstrat", rb_nq=10,
    def_alpha=0.0, def_scale=0.6, def_center=0.0, def_min_sigma=0.3,
    multi_control=False, mc_min_ess=100.0, mc_ridge=1e-6,
    stats=None,                                                          # optional dict: filled with diagnostics
):
    if getattr(neg_energy_function, "is_molecule", False):
        raise NotImplementedError("only non-molecular energies")
    with torch.no_grad():
        if t.ndim == 0:
            t = t.unsqueeze(0).repeat(len(x))
        K, (B, D) = num_mc_samples, x.shape
        dev, dt = x.device, x.dtype
        a2, b2 = get_a2_b2(t, noise_sch)
        assert torch.allclose(a2, torch.ones_like(a2)), "VE schedule only"
        h_t = b2.view(1, B, 1)
        sig = h_t.sqrt()                                   # (1,B,1)
        sig4 = sig.unsqueeze(0)                            # (1,1,B,1)
        xt4 = x.view(1, 1, B, D)

        assert K % rb_nq == 0, f"K={K} must be divisible by rb_nq={rb_nq}"
        n_A = K // rb_nq
        n_def = 0
        if def_alpha > 0:
            n_def = min(max(1, int(round(def_alpha * n_A))), n_A - 1)

        # ---- 1) nodes (all-main version, used where the defensive component is off) ----
        x_full = (xt4 + sig4 * _block(rb_mode, n_A, rb_nq, B, D, dev, dt)).reshape(K, B, D)
        x0 = x_full
        log_ratio = torch.zeros(K, B, device=dev, dtype=dt)
        active = torch.zeros(B, dtype=torch.bool, device=dev)

        # ---- 2) defensive mixture ----
        if n_def > 0:
            n_main = n_A - n_def
            x_m = xt4 + sig4 * _block(rb_mode, n_main, rb_nq, B, D, dev, dt)
            cen = torch.full((D,), float(def_center), device=dev, dtype=dt).view(1, 1, 1, D)
            x_d = cen + float(def_scale) * _block(rb_mode, n_def, rb_nq, B, D, dev, dt)
            x_act = torch.cat([x_m, x_d], dim=0).reshape(K, B, D)
            active = b2.sqrt() >= def_min_sigma
            x0 = torch.where(active.view(1, B, 1), x_act, x_full)
            a = n_def / n_A
            lN_main = _log_gauss(x0, x.unsqueeze(0), h_t.squeeze(-1), D)           # (K,B)
            lN_def = _log_gauss(x0, cen.view(1, 1, D), float(def_scale) ** 2, D)   # (K,B)
            lq = torch.logaddexp(math.log(1.0 - a) + lN_main, math.log(a) + lN_def)
            log_ratio = torch.where(active.view(1, B), lN_main - lq, torch.zeros_like(lN_main))

        log_p = torch.nan_to_num(neg_energy_function(x0).reshape(K, B),
                                 nan=_LOG_W_FLOOR, posinf=_LOG_W_FLOOR, neginf=_LOG_W_FLOOR)
        w = torch.softmax(log_p + log_ratio, dim=0).unsqueeze(-1)                  # (K,B,1)

        grad_fn = torch.func.grad(lambda v: neg_energy_function(v).sum())
        g_u = torch.nan_to_num(grad_fn(x0), nan=0.0, posinf=0.0, neginf=0.0)       # unclipped
        g_c = g_u if clipper_tsi is None else clip_scores(g_u, clipper_tsi, neg_energy_function)
        d = (x0 - x.unsqueeze(0)) / h_t

        # ---- one-control CVSI (the repo's coefficient; weights = kernel measure N(x_t, sigma^2)) ----
        c = _single_control_coef(g_u, d, torch.softmax(log_ratio, dim=0))
        c = torch.where(torch.isfinite(c), c, torch.ones_like(c))
        tilde_ct = c.view(1, B, 1)
        f = g_c - tilde_ct * (g_c - d)
        mc_use = torch.zeros(B, dtype=torch.bool, device=dev)

        # ---- 3) multi-control ----
        if multi_control:
            wk = w.squeeze(-1)                                                      # (K,B)
            ess = 1.0 / (wk ** 2).sum(0)
            ph = g_u - d                                                            # posterior score (CVSI's control)
            sg = sig.squeeze(-1)                                                    # (1,B)
            Cs = [ph]
            mu = (wk * log_p).sum(0)
            s = ((wk * (log_p - mu) ** 2).sum(0)).clamp_min(1e-12).sqrt()           # (B,)
            phi = ((log_p - mu) / s).unsqueeze(-1)
            Cs.append(phi * ph + g_u / s.view(1, B, 1))                             # logp control
            for i in range(D):                                                      # linear controls
                ci = ((x0[..., i] - x[..., i].unsqueeze(0)) / sg).unsqueeze(-1) * ph
                ci[..., i] = ci[..., i] + 1.0 / sg
                Cs.append(ci)
            X = torch.stack(Cs)                                                     # (J,K,B,D)
            J = X.shape[0]
            m = wk ** 2                                                             # squared weights ~ SNIS variance
            mC = (wk.unsqueeze(0).unsqueeze(-1) * X).sum(1)                         # (J,B,D)
            Xc = X - mC.unsqueeze(1)
            s0 = (wk.unsqueeze(-1) * g_u).sum(0)
            yc = g_u - s0.unsqueeze(0)
            G = torch.einsum("kb,jkbd,ikbd->bji", m, Xc, Xc)
            r = torch.einsum("kb,jkbd,kbd->bj", m, Xc, yc)
            tr = G.diagonal(dim1=1, dim2=2).mean(-1).clamp_min(1e-12)
            G = G + mc_ridge * tr.view(B, 1, 1) * torch.eye(J, device=dev, dtype=dt)
            beta = torch.linalg.solve_ex(G, r.unsqueeze(-1))[0].squeeze(-1)         # (B,J), no exception on singular
            f_mc = g_u - torch.einsum("bj,jkbd->kbd", beta, X)
            mc_use = (ess >= mc_min_ess) & torch.isfinite(beta).all(-1) & torch.isfinite(f_mc).all(dim=2).all(dim=0)
            f = torch.where(mc_use.view(1, B, 1), f_mc, f)
            tilde_ct = torch.where(mc_use.view(1, B, 1), beta[:, 0].view(1, B, 1), tilde_ct)

        score = (w * f).sum(dim=0)
        if stats is not None:
            stats["mc_frac"] = mc_use.float().mean()
            stats["def_frac"] = active.float().mean()
        if debug:
            return score, w, tilde_ct, g_c, d, f, log_p.unsqueeze(-1)
        if pointwise:
            return w * f, w, tilde_ct
        return score