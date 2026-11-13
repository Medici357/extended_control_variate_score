#!/usr/bin/env python
"""Estimator-level harness for Rao-Blackwellised CVSI (no training).

  python rb_harness.py --selfcheck-only --energy repo --device cuda
  python rb_harness.py --energy repo --device cuda --Ks 16 64 500 --nqs 4 10 25 --coefs prop

Changes vs. the first version:
  * --selfcheck-only exits after the self-check; --selfcheck-t picks its t (was hardcoded 0.55).
  * every variant is compared with its MATCHING baseline: prop -> repo CVSI (unweighted stats),
    post -> repo CVSI with weighted_stats=True.
  * bias2% is bias-corrected (the raw (mean-ref)^2 contains Var/R even for an unbiased estimator).
  * warns when the baseline's error is mostly a constant offset from the reference (then `ratio` is uninformative).
  * prints nA = number of independent lines/circles (RB variants) or K (iid); ESS* does not count node correlation.
"""
import argparse, csv, math, time
import torch
from dem.models.components.clipper import Clipper
from dem.models.components.noise_schedules import GeometricNoiseSchedule
from dem.models.components.cvsi import idem_score_cvsi_fn
from dem.models.components.cvsi_rb import idem_score_cvsi_rb_fn, rb_noise

BASE = "repo_cvsi"
BASE_W = "repo_cvsi_wstats"


def build_energy(kind, device):
    if kind == "repo":
        from dem.energies.gmm_energy import GMM
        e = GMM(dimensionality=2, n_mixes=40, loc_scaling=40, log_var_scaling=1.0, device=device,
                should_unnormalize=True, data_normalization_factor=50)
        return e, (lambda n: e.sample_test_set(n, normalize=True).to(device))
    from gmm_toy import GMMEnergy                      # self-contained toy GMM-40 in the normalised box
    M, D = 40, 2
    gen = torch.Generator().manual_seed(0)
    means = (torch.rand(M, D, generator=gen) - 0.5) * 2 * 0.8
    covs = (1.313 / 50) ** 2 * torch.eye(D).expand(M, D, D).clone()
    e = GMMEnergy(means.to(device), covs.to(device), torch.ones(M, device=device), normalize=False)
    return e, (lambda n: e.sample(n))


@torch.no_grad()
def reference_score(energy, xt, h, k_ref, chunk):
    """Large-K self-normalised TSI estimate (streaming log-sum-exp) used as the 'truth'."""
    B, D = xt.shape
    gfn = torch.func.grad(lambda z: energy(z).sum())
    m = torch.full((B,), -float("inf"), device=xt.device, dtype=xt.dtype)
    den = torch.zeros(B, device=xt.device, dtype=xt.dtype)
    num = torch.zeros(B, D, device=xt.device, dtype=xt.dtype)
    for _ in range(max(1, k_ref // chunk)):
        x0 = xt[None] + math.sqrt(h) * torch.randn(chunk, B, D, device=xt.device, dtype=xt.dtype)
        lw = energy(x0)
        g = torch.nan_to_num(gfn(x0))
        new_m = torch.maximum(m, lw.max(0).values)
        scale, wgt = torch.exp(m - new_m), torch.exp(lw - new_m)
        den = den * scale + wgt.sum(0)
        num = num * scale[:, None] + (wgt[..., None] * g).sum(0)
        m = new_m
    return num / den[:, None]


TAGS = {"prop": "", "post": "_post", "gpost": "_gpost"}


def make_variants(K, nqs, coefs, D, families):
    """name -> (kind, kwargs, matching_baseline_name, n_independent_units)
    families: coord | radial | rot | rotlat.   rot_q2 = antithetic pairs; rot_q4 = 4-fold rotation sampling;
    rotlat additionally rotates the radius quantile (lattice).  In D=2, rot_qN and radial_trap_qN are the same law."""
    V = {BASE: ("old", {}, BASE, K), "rb_none(=repo)": ("rb", dict(rb_mode="none"), BASE, K)}
    if "post" in coefs or "gpost" in coefs:
        V[BASE_W] = ("old", dict(weighted_stats=True), BASE_W, K)
    for q in nqs:
        if K % q or K // q < 2:
            continue
        nA = K // q
        for cw in coefs:
            tag = TAGS[cw]
            base = BASE if cw == "prop" else BASE_W
            if "coord" in families:
                V[f"coord_strat_q{q}{tag}"] = ("rb", dict(rb_mode="coord", rb_rule="strat", rb_nq=q, coef_weighting=cw), base, nA)
                V[f"coord_gh_q{q}{tag}"] = ("rb", dict(rb_mode="coord", rb_rule="gh", rb_nq=q, coef_weighting=cw), base, nA)
            if "radial" in families and D == 2:
                V[f"radial_trap_q{q}{tag}"] = ("rb", dict(rb_mode="radial", rb_rule="trap", rb_nq=q, coef_weighting=cw), base, nA)
            if "rot" in families:
                V[f"rot_q{q}{tag}"] = ("rb", dict(rb_mode="rot", rb_nq=q, coef_weighting=cw), base, nA)
            if "rotlat" in families:
                V[f"rotlat_q{q}{tag}"] = ("rb", dict(rb_mode="rot", rb_nq=q, rb_rot_radius=1, coef_weighting=cw), base, nA)
    return V


def call(kind, kw, tt, xt, energy, noise, K, clipper):
    fn = idem_score_cvsi_fn if kind == "old" else idem_score_cvsi_rb_fn
    out = fn(tt, xt, energy, noise, K, clipper_tsi=clipper, debug=True, **kw)
    return out[0], out[1]                                  # score (B,D), weights (K,B,1)


def moments(energy, xt, h, z, log_q):
    x0 = xt[None] + math.sqrt(h) * z
    u = torch.exp(energy(x0))
    m = torch.softmax(log_q, 0)
    return (m * u).sum(0), (m[..., None] * u[..., None] * x0).sum(0)      # D_hat (B,), N_hat (B,D)


@torch.no_grad()
def selfcheck(energy, noise, clipper, sample_data, dev, dtype, t=0.5, R=300, nb=40, Kb=20000, K=100, nq=10):
    h = noise.h(torch.tensor(t)).item()
    print(f"[selfcheck] t={t}  sigma={math.sqrt(h):.3g}")
    # (1) reproduction: rb_mode='none' must equal repo CVSI
    xt = sample_data(32) + math.sqrt(h) * torch.randn(32, 2, device=dev, dtype=dtype)
    tt = torch.full((32,), t, device=dev, dtype=dtype)
    torch.manual_seed(1); a = idem_score_cvsi_fn(tt, xt, energy, noise, 64, clipper_tsi=clipper)
    torch.manual_seed(1); b = idem_score_cvsi_rb_fn(tt, xt, energy, noise, 64, clipper_tsi=clipper, rb_mode="none")
    print(f"[selfcheck 1] max rel diff, rb_mode=none vs repo CVSI: {((a-b).abs().max()/a.abs().max()).item():.2e}  (expect <~1e-4)")
    # (2) the integrated numerator/denominator must be unbiased (strat, trap); gh shows its quadrature bias
    x1 = (sample_data(1) + math.sqrt(h) * torch.randn(1, 2, device=dev, dtype=dtype))
    ref = [moments(energy, x1, h, torch.randn(Kb, 1, 2, device=dev, dtype=dtype), torch.zeros(Kb, 1, device=dev, dtype=dtype))
           for _ in range(nb)]
    rD = torch.stack([r[0] for r in ref]); rN = torch.stack([r[1][..., 0] for r in ref])
    rDm, rDs, rNm, rNs = rD.mean(), rD.std() / nb ** .5, rN.mean(), rN.std() / nb ** .5
    for name, mode, rule, rm in [("coord_strat", "coord", "strat", 0), ("radial_trap", "radial", "trap", 0),
                                 ("coord_gh", "coord", "gh", 0), ("rot", "rot", "none", 0), ("rotlat", "rot", "none", 1)]:
        Ds, Ns = [], []
        for _ in range(R):
            z, lq = rb_noise(K, 1, 2, mode, rule, nq, True, dev, dtype, rot_radius=rm)
            assert torch.isfinite(z).all(), f"{name}: non-finite proposal noise"
            Dh, Nh = moments(energy, x1, h, z, lq)
            Ds.append(Dh.squeeze()); Ns.append(Nh[0, 0])
        Ds, Ns = torch.stack(Ds), torch.stack(Ns)
        zD = (Ds.mean() - rDm) / ((Ds.std() / R ** .5) ** 2 + rDs ** 2).sqrt()
        zN = (Ns.mean() - rNm) / ((Ns.std() / R ** .5) ** 2 + rNs ** 2).sqrt()
        print(f"[selfcheck 2] {name:12s} z(D_hat)={zD.item():+.2f}  z(N_hat)={zN.item():+.2f}   "
              f"rel dev D={((Ds.mean()-rDm)/rDm).item():+.2e}   (strat/trap: |z| mostly <3; gh: small nonzero = quadrature bias)")
    # (3) float32 inverse-CDF guard: many strat draws must stay finite
    z, _ = rb_noise(2_000_000 // 10 * 10, 1, 2, "coord", "strat", 10, False, dev, dtype)
    print(f"[selfcheck 3] all {z.shape[0]} strat nodes finite: {bool(torch.isfinite(z).all())}  max|z|={z.abs().max().item():.2f}")
    # (4) rotation noise in D=5 (odd D, leftover coordinate reflected): every node is N(0, I); group sums vanish when rot_radius=0
    for n4, rm in [(2, 0), (4, 0), (4, 1)]:
        Kz = 40000
        z, _ = rb_noise(Kz, 1, 5, "rot", "none", n4, False, dev, dtype, rot_radius=rm)
        zz = z.reshape(Kz // n4, n4, 5)
        print(f"[selfcheck 4] D=5 n_q={n4} rot_radius={rm}: max|node mean|={zz.mean(0).abs().max().item():.3f}  "
              f"max|node var-1|={(zz.var(0, unbiased=False) - 1).abs().max().item():.3f}  "
              f"max|group sum|={zz.sum(1).abs().max().item():.1e}   (expect ~0.01-0.03, ~0.02-0.04, and ~0 for rot_radius=0 only)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--energy", default="repo", choices=["repo", "toy"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])   # repo energy: float32 only
    ap.add_argument("--Ks", type=int, nargs="+", default=[16, 64, 500])
    ap.add_argument("--nqs", type=int, nargs="+", default=[4, 10, 25])
    ap.add_argument("--coefs", nargs="+", default=["prop"], choices=["prop", "post", "gpost"])
    ap.add_argument("--families", nargs="+", default=["radial", "rot", "rotlat"],
                    choices=["coord", "radial", "rot", "rotlat"])
    ap.add_argument("--ts", type=float, nargs="+", default=[0.2, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0])
    ap.add_argument("--B", type=int, default=128)
    ap.add_argument("--R", type=int, default=20)
    ap.add_argument("--k-ref", type=int, default=200_000)
    ap.add_argument("--chunk", type=int, default=5000)
    ap.add_argument("--sigma-min", type=float, default=1e-3)       # match configs/model/noise_schedule
    ap.add_argument("--sigma-max", type=float, default=1.0)
    ap.add_argument("--clip", type=float, default=70.0)
    ap.add_argument("--lam-eps", type=float, default=1e-3)         # BasicLambdaWeighter epsilon
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selfcheck", action="store_true", help="run the self-check, then the harness")
    ap.add_argument("--selfcheck-only", action="store_true", help="run the self-check and exit")
    ap.add_argument("--selfcheck-t", type=float, default=0.5)
    ap.add_argument("--out", default="rb_harness.csv")
    args = ap.parse_args()

    dtype = torch.float32 if args.energy == "repo" else getattr(torch, args.dtype)
    torch.set_default_dtype(dtype)
    dev = args.device
    energy, sample_data = build_energy(args.energy, dev)
    noise = GeometricNoiseSchedule(sigma_min=args.sigma_min, sigma_max=args.sigma_max)
    clipper = Clipper(True, False, max_score_norm=args.clip, min_log_reward=None)
    if args.selfcheck or args.selfcheck_only:
        selfcheck(energy, noise, clipper, sample_data, dev, dtype, t=args.selfcheck_t)
        if args.selfcheck_only:
            return

    print(f"\nR={args.R} replicates.  bias2% is bias-corrected (subtracts Var/R).  ESS* ignores node correlation.")
    rows, wsum, bases = [], {}, {}
    for ti, t in enumerate(args.ts):
        h = noise.h(torch.tensor(t)).item()
        lam = h + args.lam_eps
        torch.manual_seed(args.seed + ti)
        xt = sample_data(args.B) + math.sqrt(h) * torch.randn(args.B, 2, device=dev, dtype=dtype)
        ref = reference_score(energy, xt, h, args.k_ref, args.chunk)
        ref_norm = (ref ** 2).sum(-1).mean().item()
        tt = torch.full((args.B,), t, device=dev, dtype=dtype)
        for K in args.Ks:
            variants = make_variants(K, args.nqs, args.coefs, 2, args.families)
            st = {}
            for name, (kind, kw, base_name, nA) in variants.items():
                outs, esss = [], []
                if dev.startswith("cuda"): torch.cuda.synchronize()
                t0 = time.time()
                for r in range(args.R):
                    torch.manual_seed(args.seed * 7919 + 1000 * ti + 31 * r + K)    # same seed per replicate for every variant
                    s, w = call(kind, kw, tt, xt, energy, noise, K, clipper)
                    outs.append(s); esss.append((1.0 / (w.squeeze(-1) ** 2).sum(0)).mean())
                if dev.startswith("cuda"): torch.cuda.synchronize()
                ms = 1000 * (time.time() - t0) / args.R
                out = torch.stack(outs); mean = out.mean(0)
                pp = ((out - ref[None]) ** 2).sum(-1).mean(0)                          # (B,)
                var_pp = out.var(0, unbiased=True).sum(-1)                             # (B,)
                bias2_pp = (((mean - ref) ** 2).sum(-1) - var_pp / args.R).clamp_min(0)  # bias-corrected
                st[name] = dict(mse=pp.mean().item(), pp=pp, bias2=bias2_pp.mean().item(),
                                ess=torch.stack(esss).mean().item(), ms=ms)
                wsum[(K, name)] = wsum.get((K, name), 0.0) + lam * st[name]["mse"]
                bases[(K, name)] = base_name
            print(f"\n=== t={t}  sigma={math.sqrt(h):.3g}  loss weight={lam:.3g}  K={K} ===")
            if st[BASE]["bias2"] > 0.5 * st[BASE]["mse"]:
                print("  NOTE: >50% of the baseline's error here is a constant offset from the reference, "
                      "so `ratio` is NOT informative at this t (compare var, not MSE).")
            print(f"{'variant':26s}{'nA':>6s}{'relMSE':>10s}{'ratio':>8s}{'bias2%':>8s}{'win':>6s}{'ESS*':>8s}{'ms/call':>9s}")
            for name, (kind, kw, base_name, nA) in variants.items():
                v, b = st[name], st[base_name]
                ratio = v["mse"] / max(b["mse"], 1e-30)
                win = (v["pp"] < b["pp"]).float().mean().item()
                bshare = 100 * v["bias2"] / max(v["mse"], 1e-30)
                print(f"{name:26s}{nA:6d}{v['mse']/ref_norm:10.3g}{ratio:8.2f}{bshare:8.1f}{win:6.2f}{v['ess']:8.1f}{v['ms']:9.1f}")
                rows.append(dict(t=t, sigma=math.sqrt(h), K=K, variant=name, baseline=base_name, nA=nA,
                                 rel_mse=v["mse"] / ref_norm, ratio=ratio, bias2_pct=bshare, win=win, ess=v["ess"], ms=v["ms"]))

    print("\n##### loss-weighted MSE ratio vs MATCHING baseline (weight = h(t)+lam_eps; <1 is better; rough: t grid not uniform) #####")
    for K in args.Ks:
        names = [n for (k, n) in wsum if k == K]
        print(f"K={K}: " + "   ".join(f"{n}: {wsum[(K, n)] / wsum[(K, bases[(K, n)])]:.2f}" for n in names))
    print("\n##### loss-weighted MSE ratio vs repo_cvsi (the TRAINING DEFAULT; <1 is better) #####")
    for K in args.Ks:
        names = [n for (k, n) in wsum if k == K]
        print(f"K={K}: " + "   ".join(f"{n}: {wsum[(K, n)] / wsum[(K, BASE)]:.2f}" for n in names))
    with open(args.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()