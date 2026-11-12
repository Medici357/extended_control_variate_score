"""Rao-Blackwellised (conditional Monte Carlo) CVSI for the iDEM SNIS estimator.
VE schedule (a(t)=1), non-molecular energies only.

Estimator:  s_hat = sum_k w_k f_k,   w = softmax(log p(x0_k) + log q_k)
  q_k = quadrature weight of node k (constant for iid draws). The numerator and the denominator
  of the SNIS ratio are both integrated, never the score alone.

Fixes vs. the first version:
  * _std_normal_icdf clamps with the dtype's eps (in float32, 1 - 1e-12 == 1.0, so erfinv returned +-inf,
    which turned into NaN scores through d = (x0 - xt)/h = inf).
  * the single-control coefficient mirrors the repo exactly: no denominator clamp, and a non-finite c
    falls back to 1.0 (pure DSI), as in idem_score_cvsi_fn.
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


def rb_noise(K, B, D, mode, rule, n_q, random_frame, device, dtype):
    """Structured proposal noise.  Returns z (K,B,D) and log_q (K,B).
    K = n_A * n_q: n_A random lines (or circles), n_q nodes on each. Node index k = a * n_q + j."""
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


def idem_score_cvsi_rb_fn(
    t, x, neg_energy_function, noise_sch, num_mc_samples,
    debug=False, pointwise=False,
    divide_by_dim=False, weighted_stats=False, ct_per_molecule=False,    # accepted and ignored (drop-in signature)
    clipper_tsi=None,
    rb_mode="coord",              # none | coord | radial
    rb_rule="strat",              # strat | gh (coord)   trap (radial)
    rb_nq=10,                     # nodes per line; lines = K / rb_nq
    rb_random_frame=True,
    rb_max_sigma=None,            # use RB only where sqrt(h_t) <= this; iid draws elsewhere (same K)
    coef_weighting="prop",        # prop | post
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
        if rb_mode != "none":
            z_rb, lq_rb = rb_noise(K, B, D, rb_mode, rb_rule, rb_nq, rb_random_frame, x.device, x.dtype)
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