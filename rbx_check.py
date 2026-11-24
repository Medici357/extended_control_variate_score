"""Quick estimator-level check of the new method BEFORE training (no training involved).

1) prints the mode-coverage metric on REAL data (should be about 40 of 40 modes covered)
2) at several noise levels, compares the squared error against a big-K 'truth' for:
      base (i.i.d. CVSI) | rotstrat | rotstrat+defensive | rotstrat+multi | rotstrat+defensive+multi
   and prints each arm's error relative to base, plus the mean ESS.   Unclipped, like the harness.

  python rbx_check.py --device cuda --ts 0.5 0.7 0.85 1.0 --K 500 --B 32 --R 100
"""
import argparse
import math

import torch

from dem.models.components.cvsi_rbx import idem_score_cvsi_rbx_fn
from dem.models.components.noise_schedules import GeometricNoiseSchedule
from dem.models.dem_module_rbx import get_gmm_modes, mode_coverage
from rb_harness import build_energy, reference_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--ts", type=float, nargs="+", default=[0.5, 0.7, 0.85, 1.0])
    ap.add_argument("--K", type=int, default=500)
    ap.add_argument("--B", type=int, default=32)
    ap.add_argument("--R", type=int, default=100)
    ap.add_argument("--k-ref", type=int, default=100000)
    ap.add_argument("--chunk", type=int, default=2500)  # keep chunk*B small (large batches broke the energy before)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--scale", type=float, default=0.6)
    ap.add_argument("--dmins", type=float, default=0.3)
    ap.add_argument("--mcess", type=float, default=100.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = torch.device(args.device)
    energy, sample_data = build_energy("repo", dev)
    noise = GeometricNoiseSchedule(sigma_min=1e-3, sigma_max=1.0)

    # ---- 1) metric sanity on real data ----
    torch.manual_seed(args.seed)
    x_norm = sample_data(1000)
    x_real = energy.unnormalize(x_norm)
    locs, sd = get_gmm_modes(energy, x_real.device, x_real.dtype)
    print("mode coverage on REAL data (expect ~40 covered):", mode_coverage(x_real, locs, sd), "| modes:", tuple(locs.shape))

    arms = {
        "base": dict(rb_mode="none"),
        "rot": dict(rb_mode="rot"),
        "rotstrat": dict(rb_mode="rotstrat"),
        "rotstrat+def": dict(rb_mode="rotstrat", def_alpha=args.alpha, def_scale=args.scale, def_min_sigma=args.dmins),
        "rotstrat+mc": dict(rb_mode="rotstrat", multi_control=True, mc_min_ess=args.mcess),
        "rotstrat+def+mc": dict(rb_mode="rotstrat", def_alpha=args.alpha, def_scale=args.scale, def_min_sigma=args.dmins,
                                multi_control=True, mc_min_ess=args.mcess),
    }

    for ti, t in enumerate(args.ts):
        h = noise.h(torch.tensor(t)).item()
        torch.manual_seed(args.seed + ti)
        xt = sample_data(args.B) + math.sqrt(h) * torch.randn(args.B, 2, device=dev)
        ref = reference_score(energy, xt, h, args.k_ref, args.chunk)
        tt = torch.full((args.B,), t, device=dev)
        err = {a: torch.zeros(args.B, device=dev) for a in arms}
        ess = {a: 0.0 for a in arms}
        for _ in range(args.R):
            for a, kw in arms.items():
                out = idem_score_cvsi_rbx_fn(tt, xt, energy, noise, args.K, debug=True, clipper_tsi=None, rb_nq=10, **kw)
                score, w = out[0], out[1]
                err[a] += ((score - ref) ** 2).sum(-1)
                ess[a] += (1.0 / (w.squeeze(-1) ** 2).sum(0)).mean().item()
        print(f"\n=== t={t}  sqrt(h)={math.sqrt(h):.3g}  K={args.K} ===")
        print(f"{'arm':18s}{'mean sq. error':>16s}{'vs base':>10s}{'mean ESS':>10s}")
        base = (err["base"] / args.R).mean().item()
        for a in arms:
            e = (err[a] / args.R).mean().item()
            print(f"{a:18s}{e:16.4g}{e / base:10.3f}{ess[a] / args.R:10.1f}")


if __name__ == "__main__":
    main()