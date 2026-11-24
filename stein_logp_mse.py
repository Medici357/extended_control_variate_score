"""
Realistic test of the 'logp' Stein control at the REAL training budget (K = 500 draws).

What it does (per noise level t):
  1. Picks B test points x_t.
  2. Builds a 'truth' for the score at each point from a very large draw (K_ref, default 100000).
  3. Repeats R times: draws a fresh K=500 batch, builds several estimators from it, and measures the squared error
     against the truth.  So this measures the REAL error (variance + bias), not just a variance estimate.
  4. Prints the average error of each estimator and how it compares with 'base' (plain weighted average)
     and with 'h' (our copy of CVSI: one control, the posterior score).

Estimators ('arms'):
  base     plain self-normalised weighted average of the scores
  h        base minus a fitted multiple of the CVSI control h   (a copy of what CVSI does)
  h+logp   h plus the logp control    C = phi*h + grad(phi),  phi = standardised log p(x0)
  h+lin    h plus the two linear controls (poly1)
  h+logp+lin  all of them
Every control has zero average under the posterior (Stein identity). One coefficient is shared by both output
coordinates (trace form), fitted on the same draws with squared-weight regression (like the diagnostic).
Nothing is clipped, and the truth is unclipped too.

Run (next to rb_harness.py, from the repo, same env as the diagnostic):
  python stein_logp_mse.py --device cuda --ts 0.5 0.7 0.85 1.0 --K 500 --B 32 --R 100 --out logp_mse.csv

Caveat: the truth itself has some noise at high t (few effective draws even at K_ref). That noise adds the same
amount of error to every arm, so ratios between arms are pulled toward 1 -- real gains are at least as large.
"""
import argparse
import csv
import math

import torch

from dem.models.components.cvsi import _LOG_W_FLOOR
from dem.models.components.noise_schedules import GeometricNoiseSchedule
from rb_harness import build_energy


def draw(energy, gfn, xt, sig, K):
    B = xt.shape[0]
    eps = torch.randn(K, *xt.shape, device=xt.device)
    x0 = xt.unsqueeze(0) + sig * eps
    logp = torch.nan_to_num(energy(x0).reshape(K, B), nan=_LOG_W_FLOOR, posinf=_LOG_W_FLOOR, neginf=_LOG_W_FLOOR)
    g = torch.nan_to_num(gfn(x0), nan=0.0, posinf=0.0, neginf=0.0)
    return eps, logp, g


def make_controls(eps, logp, g, w, sig):
    """Each control C = phi * h + grad(phi) with h = posterior score; all have zero mean under the posterior."""
    D = eps.shape[-1]
    h = g - eps / sig
    C = {"h": h}
    mu = (w * logp).sum(0)
    s = (w * (logp - mu) ** 2).sum(0).clamp_min(1e-12).sqrt()  # (B,)
    phi = ((logp - mu) / s).unsqueeze(-1)
    C["logp"] = phi * h + g / s.view(1, -1, 1)  # grad of phi is g / s
    for i in range(D):  # linear controls: phi = eps_i, grad phi = e_i / sig
        c = eps[..., i].unsqueeze(-1) * h
        c[..., i] = c[..., i] + 1.0 / sig
        C[f"lin{i}"] = c
    return C


def estimate(w, g, Cs, ridge):
    """Weighted average of g minus fitted multiples of the controls (squared-weight regression, shared over D)."""
    s0 = (w.unsqueeze(-1) * g).sum(0)  # (B,D)
    if not Cs:
        return s0
    X = torch.stack(Cs)  # (J,K,B,D)
    J = X.shape[0]
    m = w ** 2
    mC = (w.unsqueeze(0).unsqueeze(-1) * X).sum(1)  # (J,B,D) weighted mean of each control
    Xc = X - mC.unsqueeze(1)
    yc = g - s0.unsqueeze(0)
    G = torch.einsum("kb,jkbd,ikbd->bji", m, Xc, Xc)
    r = torch.einsum("kb,jkbd,kbd->bj", m, Xc, yc)
    tr = G.diagonal(dim1=1, dim2=2).mean(-1).clamp_min(1e-30)
    G = G + ridge * tr.view(-1, 1, 1) * torch.eye(J, device=g.device)
    beta = torch.linalg.solve(G, r.unsqueeze(-1)).squeeze(-1)  # (B,J)
    est = s0 - torch.einsum("bj,jbd->bd", beta, mC)
    bad = ~torch.isfinite(est).all(-1, keepdim=True)
    return torch.where(bad, s0, est)


ARMS = {
    "base": [],
    "h": ["h"],
    "h+logp": ["h", "logp"],
    "h+lin": ["h", "lin0", "lin1"],
    "h+logp+lin": ["h", "logp", "lin0", "lin1"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--energy", default="repo", choices=["repo", "toy"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--ts", type=float, nargs="+", default=[0.5, 0.7, 0.85, 1.0])
    ap.add_argument("--K", type=int, default=500, help="draws per estimate (the real training budget)")
    ap.add_argument("--B", type=int, default=32, help="number of test points")
    ap.add_argument("--R", type=int, default=100, help="repeats with fresh draws")
    ap.add_argument("--K-ref", type=int, default=100000)
    ap.add_argument("--chunk", type=int, default=2500, help="keep chunk*B below ~100000 points: the energy gave wrong (flat) values for ~640000 points")
    ap.add_argument("--ridge", type=float, default=1e-6)
    ap.add_argument("--sigma-min", type=float, default=1e-3)
    ap.add_argument("--sigma-max", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="logp_mse.csv")
    args = ap.parse_args()

    dev = torch.device(args.device)
    energy, sample_data = build_energy(args.energy, dev)
    noise = GeometricNoiseSchedule(sigma_min=args.sigma_min, sigma_max=args.sigma_max)
    gfn = torch.func.grad(lambda z: energy(z).sum())
    rows = []

    for ti, t in enumerate(args.ts):
        sig = math.sqrt(noise.h(torch.tensor(t)).item())
        torch.manual_seed(args.seed + ti)
        x_data = sample_data(args.B)
        xt = x_data + sig * torch.randn_like(x_data)

        lps, gs = [], []
        for _ in range(max(1, args.K_ref // args.chunk)):
            _, lp, gg = draw(energy, gfn, xt, sig, args.chunk)
            lps.append(lp)
            gs.append(gg)
        lp, gg = torch.cat(lps), torch.cat(gs)
        wref = torch.softmax(lp, 0)
        ref = (wref.unsqueeze(-1) * gg).sum(0)  # the 'truth' (B,D)
        ess_ref = (1.0 / (wref ** 2).sum(0)).mean().item()
        del lps, gs, lp, gg, wref

        err = {a: torch.zeros(args.B, device=dev) for a in ARMS}
        ess_acc = 0.0
        for _ in range(args.R):
            eps, logp, g = draw(energy, gfn, xt, sig, args.K)
            w = torch.softmax(logp, 0)
            ess_acc += (1.0 / (w ** 2).sum(0)).mean().item()
            C = make_controls(eps, logp, g, w, sig)
            for a, names in ARMS.items():
                est = estimate(w, g, [C[n] for n in names], args.ridge)
                err[a] += ((est - ref) ** 2).sum(-1)
        for a in ARMS:
            err[a] /= args.R

        if ess_ref > 0.5 * args.K_ref and ess_acc / args.R < 0.2 * args.K:
            print("  WARNING: the truth draw has nearly flat weights but the K-draw batches do not -> the truth is broken "
                  "(energy misbehaving on a large batch). Lower --chunk and rerun.")
        print(f"\n=== t={t}  sigma={sig:.3g}  K={args.K}  mean ESS at K = {ess_acc / args.R:.1f}  (ESS of the truth draw = {ess_ref:.0f}) ===")
        print(f"{'arm':12s}{'mean sq. error':>16s}{'vs base':>10s}{'vs h':>8s}{'better than h (points)':>25s}")
        for a in ARMS:
            e = err[a].mean().item()
            vb = e / err["base"].mean().item()
            vh = e / err["h"].mean().item()
            bf = (err[a] < err["h"]).float().mean().item()
            print(f"{a:12s}{e:16.4g}{vb:10.3f}{vh:8.3f}{bf:25.2f}")
            rows.append(dict(t=t, sigma=sig, arm=a, mse=e, vs_base=vb, vs_h=vh, better_frac=bf,
                             ess_K=ess_acc / args.R, ess_ref=ess_ref))

    with open(args.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()