#!/usr/bin/env python
"""Residual-correlation diagnostic for Stein controls in CVSI (estimator level, no training).

Question answered: for the CVSI residual  r = (g - s0) - c*h   (h = posterior score, c fitted), how much of r can a
given FAMILY of Stein controls explain?   Each scalar function phi_j(x0) gives the zero-mean vector control
        C_j = phi_j * h + grad_x0 phi_j          (E_q[C_j] = 0,  q = posterior of x0 given xt)
(phi = 1 gives h itself, i.e. CVSI is the degree-0 member of this family).

Inner product (trace over the D score dimensions, weights m over the K draws):
        <a, b>_m = sum_k m_k sum_d a_kd b_kd,        m = w^2 ("snisvar", default) or w ("post"),  w = softmax(log p).
 w^2 is the right weight for the self-normalised estimator: Var(s_hat) ~ sum_k w_k^2 |f_k - s_hat|^2 (Owen, MC book 9.2).

Reported per t and per family (pooled over the B test points xt):
  heldout%   fraction of the CVSI residual energy REMOVED by fitting [h, C_1..C_J] jointly (fit on even draws, evaluated on odd
             draws and vice versa).  This is the population headroom: <= 0 means the family cannot beat CVSI even with many draws.
  insample%  same, fitted and evaluated on all draws (optimistic when J is large relative to ESS)
  better     fraction of test points where the held-out energy is below CVSI's
  pred@K     rough predicted MSE ratio vs CVSI at K=--k-target draws: (1 - heldout) * (1 + (J+1)/N) / (1 + 1/N), N = ESS_m * K / K_big.
             Rule of thumb for coefficient-estimation noise, NOT a measurement. < 1 is needed for the family to be worth using.
  zmean      zero-mean check of the controls (max over controls of the median over points of |sum_k w C| / sqrt(sum_k w^2 |C|^2));
             should be ~1-2 (a Rayleigh-type statistic for D=2).  Values >> 3 mean a bug in the control.
Per-control table: corr of each control with r AFTER residualising against h (C_perp = C - a h), and its |corr| with h.

  python stein_control_diagnostic.py --energy repo --device cuda
  python stein_control_diagnostic.py --energy repo --device cuda --ts 0.6 0.85 1.0 --families poly2 poly3 rbf logp gscore

Statistics are UNCLIPPED (like the repo's coefficient fit).  Coefficients are shared across the D output dims (trace form).
"""
import argparse
import csv
import itertools
import math

import torch

from dem.models.components.cvsi import _LOG_W_FLOOR
from dem.models.components.noise_schedules import GeometricNoiseSchedule
from rb_harness import build_energy


def ip(m, a, b):
    return (m.unsqueeze(-1) * a * b).sum((0, 2))  # (B,)


def hermite_all(x, nmax):
    """Probabilists' Hermite He_0..He_nmax of x elementwise; list of tensors shaped like x."""
    H = [torch.ones_like(x), x]
    for n in range(1, nmax):
        H.append(x * H[n] - n * H[n - 1])
    return H[: nmax + 1]


def poly_family(eps, p):
    """All Hermite monomials of total degree 1..p in eps.  Returns names, phi (J,K,B), dphi/deps (J,K,B,D)."""
    K, B, D = eps.shape
    H = hermite_all(eps, p)
    names, phis, grads = [], [], []
    for idx in itertools.product(range(p + 1), repeat=D):
        if not (1 <= sum(idx) <= p):
            continue
        comps = [H[n][..., d] for d, n in enumerate(idx)]  # each (K,B)
        phi = torch.ones_like(comps[0])
        for c in comps:
            phi = phi * c
        gl = []
        for d, n in enumerate(idx):
            if n == 0:
                gl.append(torch.zeros_like(phi))
                continue
            term = n * H[n - 1][..., d]
            for d2, c in enumerate(comps):
                if d2 != d:
                    term = term * c
            gl.append(term)
        names.append("he" + "".join(map(str, idx)))
        phis.append(phi)
        grads.append(torch.stack(gl, -1))
    assert len(names) <= 60, f"poly degree {p} in D={D} gives {len(names)} controls; lower --families degree"
    return names, torch.stack(phis), torch.stack(grads)


def rbf_family(eps, w, M, ells):
    """Gaussian bumps in eps-space, centres = posterior (weight-resampled) draws, several lengthscales."""
    K, B, D = eps.shape
    idx = torch.multinomial(w.t().contiguous(), M, replacement=True)  # (B,M)
    ar = torch.arange(B, device=eps.device).view(B, 1)
    cen = eps[idx, ar]  # (B,M,D)
    names, phis, grads = [], [], []
    for l in ells:
        for m_ in range(M):
            diff = eps - cen[:, m_].unsqueeze(0)  # (K,B,D)
            phi = torch.exp(-0.5 * (diff ** 2).sum(-1) / l ** 2)
            phis.append(phi)
            grads.append(-phi.unsqueeze(-1) * diff / l ** 2)
            names.append(f"rbf_m{m_}_l{l:g}")
    return names, torch.stack(phis), torch.stack(grads)


def hessian_rows(energy, x0):
    def f1(x):
        return energy(x.view(1, 1, -1)).reshape(())

    Hf = torch.func.jacrev(torch.func.grad(f1))
    return torch.vmap(torch.vmap(Hf))(x0)  # (K,B,D,D)


def fit_coef(X, y, m, ridge):
    """Weighted least squares of y on the J columns of X (J,K,B,D); coefficients shared over D. Returns (B,J)."""
    Xm = X * m[None, :, :, None]
    G = torch.einsum("jkbd,lkbd->bjl", Xm, X).double()
    rhs = torch.einsum("jkbd,kbd->bj", Xm, y).double()
    sc = torch.diagonal(G, dim1=-2, dim2=-1).clamp_min(1e-30).sqrt()  # (B,J)
    Gn = G / (sc.unsqueeze(-1) * sc.unsqueeze(-2)) + ridge * torch.eye(G.shape[-1], device=G.device, dtype=G.dtype)
    beta = torch.linalg.solve(Gn, (rhs / sc).unsqueeze(-1)).squeeze(-1) / sc
    return torch.nan_to_num(beta).float()


def mix(X, beta):
    return torch.einsum("bj,jkbd->kbd", beta, X)


def energy_of(m, R):
    return (m.unsqueeze(-1) * R * R).sum((0, 2))  # (B,)


def eval_set(X, y, m, folds, ridge):
    E_ho = 0.0
    for m_fit, m_ev in folds:
        E_ho = E_ho + energy_of(m_ev, y - mix(X, fit_coef(X, y, m_fit, ridge)))
    E_in = energy_of(m, y - mix(X, fit_coef(X, y, m, ridge)))
    return E_ho, E_in


def zero_mean_stat(C, w):
    S = torch.einsum("kb,jkbd->jbd", w, C).norm(dim=-1)  # (J,B)
    den = torch.einsum("kb,jkb->jb", w ** 2, (C ** 2).sum(-1)).clamp_min(1e-30).sqrt()
    return (S / den).median(dim=1).values.max().item()


def per_control(C, h, r, m):
    """corr(r, C_perp) and |corr(C, h)| per control, aggregated over points weighted by residual energy."""
    Xm = C * m[None, :, :, None]
    Chh = ip(m, h, h).clamp_min(1e-30)
    Cjh = torch.einsum("jkbd,kbd->bj", Xm, h)
    Cjj = torch.einsum("jkbd,jkbd->bj", Xm, C).clamp_min(1e-30)
    Cjr = torch.einsum("jkbd,kbd->bj", Xm, r)
    hr = ip(m, h, r)
    Rr = ip(m, r, r).clamp_min(1e-30)
    perp2 = (Cjj - Cjh ** 2 / Chh[:, None]).clamp_min(1e-30)
    perp_r = Cjr - Cjh * (hr / Chh)[:, None]
    r2 = perp_r ** 2 / (Rr[:, None] * perp2)
    ch = (Cjh ** 2 / (Cjj * Chh[:, None])).sqrt()
    wt = Rr / Rr.sum()
    return (wt[:, None] * r2).sum(0), (wt[:, None] * ch).sum(0)  # (J,), (J,)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--energy", default="repo", choices=["repo", "toy"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--ts", type=float, nargs="+", default=[0.5, 0.6, 0.7, 0.85, 1.0])
    ap.add_argument("--B", type=int, default=16)
    ap.add_argument("--K", type=int, default=40000, help="draws per point for the population (large) estimate")
    ap.add_argument("--k-target", type=int, default=500, help="draw budget for the pred@K rule of thumb")
    ap.add_argument("--families", nargs="+", default=["poly1", "poly2", "poly3", "poly4", "rbf", "logp", "gscore"])
    ap.add_argument("--rbf-M", type=int, default=8)
    ap.add_argument("--rbf-ells", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    ap.add_argument("--weights", default="snisvar", choices=["snisvar", "post"])
    ap.add_argument("--ridge", type=float, default=1e-8)
    ap.add_argument("--top", type=int, default=4)
    ap.add_argument("--sigma-min", type=float, default=1e-3)
    ap.add_argument("--sigma-max", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="stein_diag.csv")
    args = ap.parse_args()

    torch.set_default_dtype(torch.float32)
    dev = args.device
    energy, sample_data = build_energy(args.energy, dev)
    noise = GeometricNoiseSchedule(sigma_min=args.sigma_min, sigma_max=args.sigma_max)
    gfn = torch.func.grad(lambda z: energy(z).sum())
    rows = []
    K, B = args.K, args.B

    for ti, t in enumerate(args.ts):
        hh = noise.h(torch.tensor(t)).item()
        sig = math.sqrt(hh)
        torch.manual_seed(args.seed + ti)
        x_data = sample_data(B)
        D = x_data.shape[1]
        xt = x_data + sig * torch.randn(B, D, device=dev)
        eps = torch.randn(K, B, D, device=dev)
        x0 = xt.unsqueeze(0) + sig * eps
        logp = torch.nan_to_num(energy(x0).reshape(K, B), nan=_LOG_W_FLOOR, posinf=_LOG_W_FLOOR, neginf=_LOG_W_FLOOR)
        g = torch.nan_to_num(gfn(x0), nan=0.0, posinf=0.0, neginf=0.0)
        h_post = g - eps / sig  # posterior score: g - (x0 - xt)/hh
        w = torch.softmax(logp, 0)
        m = w ** 2 if args.weights == "snisvar" else w
        y = g - (w.unsqueeze(-1) * g).sum(0, keepdim=True)  # g - SNIS score estimate
        ess_w = (1.0 / (w ** 2).sum(0)).mean().item()
        ess_m = ((m.sum(0) ** 2) / (m ** 2).sum(0)).mean().item()

        par = (torch.arange(K, device=dev) % 2 == 0).float().unsqueeze(-1)
        folds = [(m * par, m * (1 - par)), (m * (1 - par), m * par)]
        X1 = h_post.unsqueeze(0)
        E0 = energy_of(m, y)
        Ec_ho, Ec_in = eval_set(X1, y, m, folds, args.ridge)
        c_full = fit_coef(X1, y, m, args.ridge)[:, 0]
        r = y - c_full.view(1, B, 1) * h_post
        zh = zero_mean_stat(h_post.unsqueeze(0), w)
        print(f"\n=== t={t}  sigma={sig:.3g}  B={B}  K_big={K}  ESS_w={ess_w:.1f}  ESS_m={ess_m:.1f}  weights={args.weights} ===")
        print(f"CVSI (one control h) removes {100 * (1 - Ec_ho.sum() / E0.sum()).item():.1f}% of the raw score-estimator energy (held-out);"
              f"  zero-mean check of h: {zh:.2f}")
        print(f"{'family':10s}{'J':>4s}{'heldout%':>10s}{'insample%':>11s}{'better':>8s}{'pred@K':>9s}{'zmean':>8s}")

        N_target = max(ess_m * args.k_target / K, 1.0)
        for fam in args.families:
            try:
                if fam.startswith("poly"):
                    names, phi, ge = poly_family(eps, int(fam[4:]))
                    gx = ge / sig
                elif fam == "rbf":
                    names, phi, ge = rbf_family(eps, w, args.rbf_M, args.rbf_ells)
                    gx = ge / sig
                elif fam == "logp":
                    mu = (w * logp).sum(0)
                    s = ((w * (logp - mu) ** 2).sum(0)).clamp_min(1e-12).sqrt()  # (B,)
                    names = ["logp"]
                    phi = ((logp - mu) / s).unsqueeze(0)
                    gx = (g / s.view(1, B, 1)).unsqueeze(0)
                elif fam == "gscore":
                    Hs = hessian_rows(energy, x0)  # (K,B,D,D)
                    names, phis, gxs = [], [], []
                    for i in range(D):
                        mu = (w * g[..., i]).sum(0)
                        s = ((w * (g[..., i] - mu) ** 2).sum(0)).clamp_min(1e-12).sqrt()
                        phis.append(g[..., i] / s)
                        gxs.append(torch.nan_to_num(Hs[:, :, i, :]) / s.view(1, B, 1))
                        names.append(f"g[{i}]")
                    phi, gx = torch.stack(phis), torch.stack(gxs)
                else:
                    print(f"{fam:10s} unknown family, skipped")
                    continue
            except Exception as e:  # e.g. energy not vmap-able for gscore
                print(f"{fam:10s} skipped ({type(e).__name__}: {str(e)[:80]})")
                continue
            C = phi.unsqueeze(-1) * h_post.unsqueeze(0) + gx  # (J,K,B,D)
            J = C.shape[0]
            z = zero_mean_stat(C, w)
            X = torch.cat([X1, C], 0)
            Ef_ho, Ef_in = eval_set(X, y, m, folds, args.ridge)
            ratio = (Ef_ho.sum() / Ec_ho.sum()).item()
            gain_ho, gain_in = 100 * (1 - ratio), 100 * (1 - (Ef_in.sum() / Ec_in.sum()).item())
            better = (Ef_ho < Ec_ho).float().mean().item()
            pred = ratio * (1 + (J + 1) / N_target) / (1 + 1 / N_target)
            print(f"{fam:10s}{J:4d}{gain_ho:10.1f}{gain_in:11.1f}{better:8.2f}{pred:9.2f}{z:8.2f}")
            rows.append(dict(t=t, sigma=sig, family=fam, J=J, heldout_pct=gain_ho, insample_pct=gain_in,
                             better_frac=better, pred_ratio=pred, zero_mean_z=z, ess_w=ess_w, ess_m=ess_m))
            if args.top > 0:
                r2, ch = per_control(C, h_post, r, m)
                top = torch.argsort(r2, descending=True)[: args.top]
                print("           top single controls (R2 vs residual after removing h | |corr with h|): " +
                      "  ".join(f"{names[i]}:{r2[i].item():.3f}|{ch[i].item():.2f}" for i in top.tolist()))

    if rows:
        with open(args.out, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()