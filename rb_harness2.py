#!/usr/bin/env python
"""rb_harness2.py -- estimator-level comparison of CVSI variants (NO training).

What it does
------------
At many noise levels t it takes noisy query points x_t, builds a very accurate reference for the true
noised score, then runs every estimator variant R times (fresh randomness, same points) and reports

    variance  (random noise of the estimator)      -> what variance-reduction methods are supposed to cut
    bias      (systematic error vs the reference)  -> NOT fixed by variance reduction (SNIS ratio bias, clipping, ...)
    MSE       (variance + bias^2)                  -> distance to the truth

for each variant relative to its baseline, with 95% bootstrap confidence intervals and a plain verdict.

Variant families (choose with --families)
    rb     cvsi_rb.py   : the code behind your reported results (radial / rot / rotlat, optional coord)
    rbx    cvsi_rbx.py  : rotstrat rings, defensive mixture, multi-control (needs the file to exist)
    local  (this file)  : clean ablations that separate "sampler" from "coefficient rule" from "extra controls"

Examples (you run them)
    python rb_harness2.py --selfcheck-only --device cuda
    python rb_harness2.py --device cuda --Ks 64 500
    python rb_harness2.py --device cuda --Ks 500 --clip 70 --detail     # match training clipping
Outputs: <prefix>_report.txt (everything printed), <prefix>.csv (long format), <prefix>_K<k>.png (plots).
"""
import argparse
import csv
import math
import re
import time

import torch

from dem.models.components.clipper import Clipper
from dem.models.components.cvsi import _LOG_W_FLOOR, get_a2_b2, idem_score_cvsi_fn
from dem.models.components.cvsi_rb import idem_score_cvsi_rb_fn, rb_noise
from dem.models.components.noise_schedules import GeometricNoiseSchedule

try:
    from dem.models.components.cvsi_rbx import _block as _rbx_block
    from dem.models.components.cvsi_rbx import idem_score_cvsi_rbx_fn

    HAVE_RBX = True
    _RBX_ERR = ""
except Exception as _err:  # rbx file missing/broken: the other families still run
    HAVE_RBX = False
    _RBX_ERR = repr(_err)

SYM = {"better": "✓", "WORSE": "✗", "tie": "~"}

LEGEND = """
HOW TO READ THIS REPORT (plain language)
  query point   a noisy location x_t where we estimate the score (= gradient of the log-density of the noised
                data). B of them are drawn at every noise level.
  reference     a very accurate estimate of the true score at each query point (huge sample, computed twice
                independently so we know how accurate the reference itself is). It plays the role of "truth".
  replicate     re-running an estimator R times with fresh randomness at the SAME query points.
  variance      how much the answer jumps around between replicates  (the estimator's random noise).
  bias          how far the average answer is from the reference      (systematic error; variance reduction
                does not remove it; it mostly comes from the 1/ESS bias of self-normalised importance sampling
                and, if --clip is on, from clipping).
  MSE           variance + bias^2 = typical squared distance to the truth.
  var ratio     variance of a method / variance of its baseline.  BELOW 1 = less noise.  The brackets are a 95%
                bootstrap confidence interval (we resample query points).  Verdict symbols:
                  ✓ whole interval is below 1  -> real improvement
                  ✗ whole interval is above 1  -> real degradation
                  ~ interval contains 1        -> cannot be told apart from the baseline
  train-wtd     average over noise levels using the training loss weight lambda(t)=h(t)+eps, so it is the
                number that matters for the training objective (it is dominated by high noise).
  low/mid/high  equal-weight average over the noise levels inside a band of sigma (so each level counts the
                same, in units of the squared score size). Bands are set by --bands.
  x cost        var ratio multiplied by the time ratio. Below 1 means it still wins at equal compute time.
  ESS           effective sample size = 1 / sum(w^2): how many of the K points really carry weight.
  units         number of independent random draws (rings / lines / orbits). Points inside a ring are
                dependent, so more units = more independent information. iid has units = K.
""".strip("\n")


# ----------------------------------------------------------------------------------------------- reporting
class Report:
    def __init__(self):
        self.lines = []

    def p(self, *a):
        line = " ".join(str(x) for x in a)
        print(line, flush=True)
        self.lines.append(line)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.lines) + "\n")


def verdict(lo, hi):
    return "better" if hi < 1.0 else ("WORSE" if lo > 1.0 else "tie")


def cell(r, lo, hi):
    return f"{r:5.2f}{SYM[verdict(lo, hi)]}"


# ----------------------------------------------------------------------------------------------- energy / reference
def build_energy(kind, device):
    if kind == "repo":
        from dem.energies.gmm_energy import GMM

        e = GMM(dimensionality=2, n_mixes=40, loc_scaling=40, log_var_scaling=1.0, device=device,
                should_unnormalize=True, data_normalization_factor=50)
        return e, (lambda n: e.sample_test_set(n, normalize=True).to(device))
    from gmm_toy import GMMEnergy

    M, D = 40, 2
    gen = torch.Generator().manual_seed(0)
    means = (torch.rand(M, D, generator=gen) - 0.5) * 2 * 0.8
    covs = (1.313 / 50) ** 2 * torch.eye(D).expand(M, D, D).clone()
    e = GMMEnergy(means.to(device), covs.to(device), torch.ones(M, device=device), normalize=False)
    return e, (lambda n: e.sample(n))


@torch.no_grad()
def reference_score(energy, xt, h, n_draws, chunk):
    """Large-sample self-normalised TSI estimate (streaming log-sum-exp) used as the 'truth'."""
    B, D = xt.shape
    gfn = torch.func.grad(lambda z: energy(z).sum())
    m = torch.full((B,), -float("inf"), device=xt.device, dtype=xt.dtype)
    den = torch.zeros(B, device=xt.device, dtype=xt.dtype)
    num = torch.zeros(B, D, device=xt.device, dtype=xt.dtype)
    for _ in range(max(1, n_draws // chunk)):
        x0 = xt[None] + math.sqrt(h) * torch.randn(chunk, B, D, device=xt.device, dtype=xt.dtype)
        lw = torch.nan_to_num(energy(x0).reshape(chunk, B), nan=_LOG_W_FLOOR, posinf=_LOG_W_FLOOR,
                              neginf=_LOG_W_FLOOR)
        g = torch.nan_to_num(gfn(x0))
        new_m = torch.maximum(m, lw.max(0).values)
        scale, wgt = torch.exp(m - new_m), torch.exp(lw - new_m)
        den = den * scale + wgt.sum(0)
        num = num * scale[:, None] + (wgt[..., None] * g).sum(0)
        m = new_m
    return num / den[:, None]


# ----------------------------------------------------------------------------------------------- noise samplers
_SAMPLER_RE = re.compile(r"^([a-z]+)(\d+)$")


def sampler_ok(sampler, K, D):
    if sampler == "iid":
        return True
    m = _SAMPLER_RE.match(sampler)
    if not m:
        return False
    kind, q = m.group(1), int(m.group(2))
    if q < 2 or K % q or K // q < 2:
        return False
    if kind == "rotstrat":
        return HAVE_RBX and D == 2
    return kind in ("rot", "rotlat")


def sampler_units(sampler, K):
    if sampler == "iid":
        return K
    return K // int(_SAMPLER_RE.match(sampler).group(2))


SAMPLER_TEXT = {
    "iid": "plain independent Gaussian draws",
    "rot": "rotation sampling (one Gaussian draw per group rotated into q equally spaced copies; in 2D a ring)",
    "rotlat": "rotation sampling + radius-quantile lattice (radii of a group spread over q levels)",
    "rotstrat": "rings whose RADIUS is stratified across groups (rbx _polar_strat_rot)",
}


def draw_noise(sampler, K, B, D, dev, dt):
    """Returns z (K,B,D) and log_q (K,B). Pooled over all K points the noise is N(0, I)."""
    if sampler == "iid":
        return torch.randn(K, B, D, device=dev, dtype=dt), torch.zeros(K, B, device=dev, dtype=dt)
    kind, q = _SAMPLER_RE.match(sampler).group(1), int(_SAMPLER_RE.match(sampler).group(2))
    if kind == "rot":
        return rb_noise(K, B, D, "rot", "none", q, False, dev, dt, rot_radius=0)
    if kind == "rotlat":
        return rb_noise(K, B, D, "rot", "none", q, False, dev, dt, rot_radius=1)
    if kind == "rotstrat":
        z = _rbx_block("rotstrat", K // q, q, B, D, dev, dt).reshape(K, B, D)
        return z, torch.zeros(K, B, device=dev, dtype=dt)
    raise ValueError(sampler)


# ----------------------------------------------------------------------------------------------- local Stein estimator
def stein_controls(names, x0, x, g_u, d, wk, log_p, sg):
    """Zero-mean (under the posterior) Stein controls C_phi = phi*h + grad(phi), h = g - d.  Each is (K,B,D)."""
    ph = g_u - d
    out = []
    if "h" in names:
        out.append(ph)
    if "logp" in names:
        mu = (wk * log_p).sum(0)
        s = (wk * (log_p - mu) ** 2).sum(0).clamp_min(1e-12).sqrt()  # (B,)
        phi = ((log_p - mu) / s).unsqueeze(-1)
        out.append(phi * ph + g_u / s.view(1, -1, 1))
    if "lin" in names:
        for i in range(x0.shape[-1]):
            ci = ((x0[..., i] - x[..., i].unsqueeze(0)) / sg).unsqueeze(-1) * ph
            ci[..., i] = ci[..., i] + 1.0 / sg
            out.append(ci)
    return out


@torch.no_grad()
def _core(tt, xt, energy, noise, K, sampler):
    B, D = xt.shape
    _, b2 = get_a2_b2(tt, noise)
    h = b2.view(1, B, 1)
    z, log_q = draw_noise(sampler, K, B, D, xt.device, xt.dtype)
    x0 = xt.unsqueeze(0) + z * h.sqrt()
    log_p = torch.nan_to_num(energy(x0).reshape(K, B), nan=_LOG_W_FLOOR, posinf=_LOG_W_FLOOR,
                             neginf=_LOG_W_FLOOR)
    wk = torch.softmax(log_p + log_q, dim=0)
    gfn = torch.func.grad(lambda v: energy(v).sum())
    g_u = torch.nan_to_num(gfn(x0), nan=0.0, posinf=0.0, neginf=0.0)
    d = (x0 - xt.unsqueeze(0)) / h
    return x0, log_p, log_q, wk, g_u, d, b2.sqrt().view(1, B)


@torch.no_grad()
def local_stein_estimator(tt, xt, energy, noise, K, sampler, controls, weighting, ridge=1e-6):
    """CVSI-style estimator with a chosen sampler, control set and coefficient rule. Never clips.
    weighting 'ols': proposal-weighted least squares (with controls=('h',) and sampler 'iid' this IS repo CVSI).
    weighting 'w2' : least squares weighted by squared SNIS weights (the rule used in rbx multi-control)."""
    B, D = xt.shape
    x0, log_p, log_q, wk, g_u, d, sg = _core(tt, xt, energy, noise, K, sampler)
    X = torch.stack(stein_controls(controls, x0, xt, g_u, d, wk, log_p, sg))  # (J,K,B,D)
    J = X.shape[0]
    if weighting == "ols":
        m = torch.softmax(log_q, 0)
        cw = m
    elif weighting == "w2":
        m = wk ** 2
        cw = wk
    else:
        raise ValueError(weighting)
    mC = (cw.view(1, K, B, 1) * X).sum(1)
    Xc = X - mC.unsqueeze(1)
    s0 = (cw.unsqueeze(-1) * g_u).sum(0)
    yc = g_u - s0.unsqueeze(0)
    G = torch.einsum("kb,jkbd,ikbd->bji", m, Xc, Xc)
    r = torch.einsum("kb,jkbd,kbd->bj", m, Xc, yc)
    tr = G.diagonal(dim1=1, dim2=2).mean(-1).clamp_min(1e-12)
    G = G + ridge * tr.view(B, 1, 1) * torch.eye(J, device=xt.device, dtype=xt.dtype)
    beta = torch.nan_to_num(torch.linalg.solve_ex(G, r.unsqueeze(-1))[0].squeeze(-1), nan=0.0, posinf=0.0,
                            neginf=0.0)
    f = g_u - torch.einsum("bj,jkbd->kbd", beta, X)
    return (wk.unsqueeze(-1) * f).sum(0), wk.unsqueeze(-1)


# ----------------------------------------------------------------------------------------------- variants
class Variant:
    def __init__(self, name, group, run, base, units, text):
        self.name, self.group, self.run, self.base, self.units, self.text = name, group, run, base, units, text


def run_repo(weighted):
    def run(tt, xt, energy, noise, K, clipper):
        o = idem_score_cvsi_fn(tt, xt, energy, noise, K, clipper_tsi=clipper, debug=True, weighted_stats=weighted)
        return o[0], o[1]
    return run


def run_rb(**kw):
    def run(tt, xt, energy, noise, K, clipper):
        o = idem_score_cvsi_rb_fn(tt, xt, energy, noise, K, clipper_tsi=clipper, debug=True, **kw)
        return o[0], o[1]
    return run


def run_rbx(**kw):
    def run(tt, xt, energy, noise, K, clipper):
        o = idem_score_cvsi_rbx_fn(tt, xt, energy, noise, K, clipper_tsi=clipper, debug=True, **kw)
        return o[0], o[1]
    return run


def run_local(sampler, controls, weighting):
    def run(tt, xt, energy, noise, K, clipper):
        return local_stein_estimator(tt, xt, energy, noise, K, sampler, controls, weighting)
    return run


G0, G1, G2, G3, G4 = "0 baseline", "1 rb (reported results)", "2 rbx sampling", "3 rbx controls/mixture", \
    "4 local ablations"


def build_variants(K, D, args):
    vs = []

    def add(*a):
        vs.append(Variant(*a))

    fam, nqs = set(args.families), [q for q in args.nqs if K % q == 0 and K // q >= 2]
    add("repo_cvsi", G0, run_repo(False), "repo_cvsi", K,
        "original CVSI: iid noise, ONE control, plain coefficient. This is what training uses today.")
    if any(c != "prop" for c in args.coefs):
        add("repo_cvsi_w", G0, run_repo(True), "repo_cvsi_w", K,
            "original CVSI with SNIS-weighted coefficient statistics (baseline for post/gpost rows)")
    if "rb" in fam or "coord" in fam:
        add("rb_none(=repo)", G1, run_rb(rb_mode="none"), "repo_cvsi", K,
            "SANITY: cvsi_rb with no structure; must match repo_cvsi")
        for q in nqs:
            for cw in args.coefs:
                tag = "" if cw == "prop" else "_" + cw
                base = "repo_cvsi" if cw == "prop" else "repo_cvsi_w"
                ctext = {"prop": "plain coefficient", "post": "SNIS-weighted coefficient",
                         "gpost": "group-level SNIS-weighted coefficient"}[cw]
                if "rb" in fam:
                    if D == 2:
                        add(f"radial_trap_q{q}{tag}", G1, run_rb(rb_mode="radial", rb_rule="trap", rb_nq=q,
                                                                  coef_weighting=cw), base, K // q,
                            f"ring of {q} equally spaced angles, random radius (your method B); {ctext}")
                    add(f"rot_q{q}{tag}", G1, run_rb(rb_mode="rot", rb_nq=q, coef_weighting=cw), base, K // q,
                        f"rotation sampling, {q} copies per group (in 2D identical to radial_trap); {ctext}")
                    add(f"rotlat_q{q}{tag}", G1, run_rb(rb_mode="rot", rb_nq=q, rb_rot_radius=1,
                                                         coef_weighting=cw), base, K // q,
                        f"rotation + radius lattice (the only rb mode that spreads radii); {ctext}")
                if "coord" in fam:
                    for rule in ("strat", "gh"):
                        add(f"coord_{rule}_q{q}{tag}", G1, run_rb(rb_mode="coord", rb_rule=rule, rb_nq=q,
                                                                   coef_weighting=cw), base, K // q,
                            f"{q} quantile nodes along a random line ({rule}); {ctext}")
    if "rbx" in fam:
        if not HAVE_RBX:
            vs.append(None)  # marker handled by caller
        else:
            mm = args.mc_min_ess
            for q in nqs:
                if D == 2:
                    add(f"rbx_rotstrat_q{q}", G2, run_rbx(rb_mode="rotstrat", rb_nq=q), "repo_cvsi", K // q,
                        f"rings with radius STRATIFIED across the {K // q} groups, {q} angles each; one control")
                    add(f"rbx_rotstrat_q{q}+mc", G3, run_rbx(rb_mode="rotstrat", rb_nq=q, multi_control=True,
                                                             mc_min_ess=mm), "repo_cvsi", K // q,
                        f"rotstrat + multi-control (h, logp, linear); controls only used where ESS >= {mm:g}")
                    for a in args.def_alphas:
                        add(f"rbx_rotstrat_q{q}+def{a:g}", G3, run_rbx(rb_mode="rotstrat", rb_nq=q, def_alpha=a),
                            "repo_cvsi", K // q,
                            f"rotstrat + defensive mixture ({a:g} of groups drawn from a broad Gaussian; only "
                            "acts where sigma >= def_min_sigma=0.3)")
                add(f"rbx_rot_q{q}+mc", G3, run_rbx(rb_mode="rot", rb_nq=q, multi_control=True, mc_min_ess=mm),
                    "repo_cvsi", K // q, f"plain rotation + multi-control (isolates the controls from rotstrat)")
            add("rbx_iid+mc", G3, run_rbx(rb_mode="none", rb_nq=1, multi_control=True, mc_min_ess=mm),
                "repo_cvsi", K, "iid noise + multi-control (isolates the controls from any sampler change)")
    if "local" in fam:
        add("loc_iid_h_ols", G4, run_local("iid", ("h",), "ols"), "repo_cvsi", K,
            "SANITY: must match repo_cvsi when --clip 0 (same estimator written inside this file)")
        for s in args.local_samplers:
            if not sampler_ok(s, K, D):
                continue
            kind = "iid" if s == "iid" else _SAMPLER_RE.match(s).group(1)
            for cs in args.local_controls:
                add(f"loc_{s}_{cs}", G4, run_local(s, tuple(cs.split("+")), "w2"), "repo_cvsi", sampler_units(s, K),
                    f"{SAMPLER_TEXT[kind]}; controls={cs}; coefficient by squared-SNIS-weight least squares; "
                    "never clips")
    return [v for v in vs if v is not None]


# ----------------------------------------------------------------------------------------------- self-checks
def moments(energy, xt, h, z, log_q):
    x0 = xt[None] + math.sqrt(h) * z
    u = torch.exp(energy(x0).reshape(z.shape[0], z.shape[1]))
    m = torch.softmax(log_q, 0)
    return (m * u).sum(0), (m[..., None] * u[..., None] * x0).sum(0)


@torch.no_grad()
def selfcheck(rep, energy, noise, sample_data, dev, dt, t):
    h = noise.h(torch.tensor(t)).item()
    rep.p(f"[selfcheck] t={t}  sigma={math.sqrt(h):.3g}")
    xt = sample_data(32) + math.sqrt(h) * torch.randn(32, 2, device=dev, dtype=dt)
    tt = torch.full((32,), t, device=dev, dtype=dt)
    torch.manual_seed(1)
    a = idem_score_cvsi_fn(tt, xt, energy, noise, 64, clipper_tsi=None)
    torch.manual_seed(1)
    b = idem_score_cvsi_rb_fn(tt, xt, energy, noise, 64, clipper_tsi=None, rb_mode="none")
    torch.manual_seed(1)
    c, _ = local_stein_estimator(tt, xt, energy, noise, 64, "iid", ("h",), "ols")
    den = a.abs().max().clamp_min(1e-30)
    rep.p(f"[1] same seed, no clip: max relative difference  repo vs cvsi_rb(none) = {((a - b).abs().max() / den).item():.1e}"
          f"   repo vs local(iid,h,ols) = {((a - c).abs().max() / den).item():.1e}   (expect < ~1e-4 for both)")

    # (2) each noise sampler must give unbiased unnormalised moments (pooled over all K points)
    nb, Kb, R, K = 40, 20000, 300, 100
    x1 = sample_data(1) + math.sqrt(h) * torch.randn(1, 2, device=dev, dtype=dt)
    ref = [moments(energy, x1, h, *draw_noise("iid", Kb, 1, 2, dev, dt)) for _ in range(nb)]
    rD = torch.stack([r[0] for r in ref]).squeeze()
    rN = torch.stack([r[1][0, 0] for r in ref])
    for s in ("iid", "rot4", "rotlat4", "rotstrat10"):
        if not sampler_ok(s, K, 2):
            rep.p(f"[2] {s:11s} skipped (not available here)")
            continue
        Ds, Ns = [], []
        for _ in range(R):
            Dh, Nh = moments(energy, x1, h, *draw_noise(s, K, 1, 2, dev, dt))
            Ds.append(Dh.squeeze())
            Ns.append(Nh[0, 0])
        Ds, Ns = torch.stack(Ds), torch.stack(Ns)
        zD = (Ds.mean() - rD.mean()) / ((Ds.std() / R ** .5) ** 2 + (rD.std() / nb ** .5) ** 2).sqrt()
        zN = (Ns.mean() - rN.mean()) / ((Ns.std() / R ** .5) ** 2 + (rN.std() / nb ** .5) ** 2).sqrt()
        rep.p(f"[2] {s:11s} z-score of mean(D_hat) = {zD.item():+.2f}, mean(N_hat) = {zN.item():+.2f}   "
              "(unbiased samplers: |z| mostly < 3)")

    # (3) Stein controls must have mean zero under the posterior (checks the control formulas)
    Kc, Rc = 2000, 200
    tt1 = torch.full((1,), t, device=dev, dtype=dt)
    ms = []
    for _ in range(Rc):
        x0, log_p, log_q, wk, g_u, d, sg = _core(tt1, x1, energy, noise, Kc, "iid")
        X = torch.stack(stein_controls(("h", "logp", "lin"), x0, x1, g_u, d, wk, log_p, sg))
        ms.append((wk.view(1, Kc, 1, 1) * X).sum(1))
    ms = torch.stack(ms).squeeze(2)  # (R,J,D)
    z = ms.mean(0) / (ms.std(0) / Rc ** .5).clamp_min(1e-30)
    labels = ["h", "logp"] + [f"lin{i}" for i in range(2)]
    rep.p("[3] Stein controls, mean of control under SNIS weights (should be ~0): max |z| per control: "
          + ", ".join(f"{lb}={z[j].abs().max().item():.2f}" for j, lb in enumerate(labels))
          + "   (|z| < ~3 expected; 'logp' may be slightly off because its scaling uses the same samples)")


# ----------------------------------------------------------------------------------------------- bootstrap aggregation
def make_agg(RES, IDX, nboot):
    qs = torch.tensor([0.025, 0.975])

    def agg(K, name, base, tis, wts, key):
        num = torch.zeros(nboot)
        den = torch.zeros(nboot)
        pn = pd = 0.0
        for ti, wt in zip(tis, wts):
            a, b, iv = RES[(K, name)][ti][key], RES[(K, base)][ti][key], IDX[ti]
            num += wt * a[iv].mean(1)
            den += wt * b[iv].mean(1)
            pn += wt * a.mean().item()
            pd += wt * b.mean().item()
        r = num / den.clamp_min(1e-30)
        lo, hi = torch.quantile(r, qs).tolist()
        return pn / max(pd, 1e-30), lo, hi

    return agg


# ----------------------------------------------------------------------------------------------- main
def parse():
    ap = argparse.ArgumentParser(description="Estimator-level CVSI harness with explained output")
    ap.add_argument("--energy", default="repo", choices=["repo", "toy"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    ap.add_argument("--Ks", type=int, nargs="+", default=[64, 500])
    ap.add_argument("--nqs", type=int, nargs="+", default=[4, 10])
    ap.add_argument("--families", nargs="+", default=["rb", "rbx", "local"], choices=["rb", "rbx", "local", "coord"])
    ap.add_argument("--coefs", nargs="+", default=["prop"], choices=["prop", "post", "gpost"])
    ap.add_argument("--def-alphas", type=float, nargs="*", default=[0.2])
    ap.add_argument("--local-samplers", nargs="+", default=["iid", "rot4", "rotstrat10"])
    ap.add_argument("--local-controls", nargs="+", default=["h", "h+logp", "h+lin", "h+logp+lin"])
    ap.add_argument("--mc-min-ess", type=float, default=100.0)
    ap.add_argument("--n-t", type=int, default=10, help="number of equally spaced noise levels (midpoints of [0,1])")
    ap.add_argument("--ts", type=float, nargs="+", default=None)
    ap.add_argument("--B", type=int, default=128, help="query points per noise level")
    ap.add_argument("--R", type=int, default=30, help="replicates per estimator")
    ap.add_argument("--k-ref", type=int, default=200_000)
    ap.add_argument("--chunk", type=int, default=5000)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--bands", type=float, nargs=2, default=[0.03, 0.3], help="sigma edges: low | mid | high")
    ap.add_argument("--clip", type=float, default=0.0, help="TSI clip norm; 0 = off (pure estimator comparison)")
    ap.add_argument("--sigma-min", type=float, default=1e-3)
    ap.add_argument("--sigma-max", type=float, default=1.0)
    ap.add_argument("--lam-eps", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--selfcheck-only", action="store_true")
    ap.add_argument("--selfcheck-t", type=float, default=0.5)
    ap.add_argument("--detail", action="store_true", help="also print one table per noise level")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--out-prefix", default="rb_harness2")
    return ap.parse_args()


def main():
    args = parse()
    rep = Report()
    dtype = torch.float32 if args.energy == "repo" else getattr(torch, args.dtype)
    torch.set_default_dtype(dtype)
    dev = args.device
    energy, sample_data = build_energy(args.energy, dev)
    noise = GeometricNoiseSchedule(sigma_min=args.sigma_min, sigma_max=args.sigma_max)
    clipper = None if args.clip <= 0 else Clipper(True, False, max_score_norm=args.clip, min_log_reward=None)
    if args.selfcheck or args.selfcheck_only:
        selfcheck(rep, energy, noise, sample_data, dev, dtype, args.selfcheck_t)
        if args.selfcheck_only:
            rep.save(args.out_prefix + "_selfcheck.txt")
            return

    ts = args.ts or [(i + 0.5) / args.n_t for i in range(args.n_t)]
    D = sample_data(2).shape[1]
    VARS = {K: build_variants(K, D, args) for K in args.Ks}
    if "rbx" in args.families and not HAVE_RBX:
        rep.p(f"NOTE: rbx family skipped, could not import cvsi_rbx ({_RBX_ERR})")

    rep.p(LEGEND)
    rep.p("")
    rep.p("SETTINGS")
    rep.p(f"  energy={args.energy}  noise levels t={['%.3g' % t for t in ts]}  query points B={args.B}  replicates R={args.R}")
    rep.p(f"  reference = {args.k_ref} draws (two independent halves)   bootstrap resamples={args.n_boot}")
    rep.p("  TSI clipping: " + ("OFF -> compares pure estimator quality (recommended for variance work)." if clipper is None
          else f"ON at norm {args.clip:g} (matches training). NOTE: rbx multi-control and all loc_* rows never clip, so "
               "they are NOT directly comparable under clipping."))
    rep.p("")
    rep.p("VARIANT GUIDE (names are the row labels used below)")
    seen = set()
    for K in args.Ks:
        for v in VARS[K]:
            key = re.sub(r"\d+", "#", v.name)
            if key in seen:
                continue
            seen.add(key)
            rep.p(f"  [{v.group}] {v.name}: {v.text}")

    # ---------------------------------------------------------------- run
    RES, META, IDX = {}, {}, {}
    gboot = torch.Generator().manual_seed(args.seed)
    for ti, t in enumerate(ts):
        h = noise.h(torch.tensor(t)).item()
        torch.manual_seed(args.seed + ti)
        xt = sample_data(args.B) + math.sqrt(h) * torch.randn(args.B, D, device=dev, dtype=dtype)
        ra = reference_score(energy, xt, h, args.k_ref // 2, args.chunk)
        rb2 = reference_score(energy, xt, h, args.k_ref // 2, args.chunk)
        ref = 0.5 * (ra + rb2)
        refvar = ((ra - rb2) ** 2).sum(-1) / 4.0  # variance of the averaged reference, per point
        META[ti] = dict(t=t, sigma=math.sqrt(h), lam=h + args.lam_eps, ref_norm=(ref ** 2).sum(-1).mean().item(),
                        refvar=refvar.mean().item())
        IDX[ti] = torch.randint(args.B, (args.n_boot, args.B), generator=gboot)
        tt = torch.full((args.B,), t, device=dev, dtype=dtype)
        print(f"[run] t={t:.3g} sigma={math.sqrt(h):.3g}  reference noise floor = "
              f"{100 * META[ti]['refvar'] / META[ti]['ref_norm']:.3f}% of |score|^2", flush=True)
        for K in args.Ks:
            for v in VARS[K]:
                seed0 = args.seed * 7919 + 1000 * ti + K
                torch.manual_seed(seed0 - 1)
                v.run(tt, xt, energy, noise, K, clipper)  # warm-up, not counted
                if dev.startswith("cuda"):
                    torch.cuda.synchronize()
                t0 = time.time()
                outs, esss = [], []
                for r in range(args.R):
                    torch.manual_seed(seed0 + 31 * r)
                    s, w = v.run(tt, xt, energy, noise, K, clipper)
                    outs.append(s)
                    esss.append((1.0 / (w.squeeze(-1) ** 2).sum(0)).mean())
                if dev.startswith("cuda"):
                    torch.cuda.synchronize()
                ms = 1000 * (time.time() - t0) / args.R
                o = torch.stack(outs)
                var_pp = o.var(0, unbiased=True).sum(-1)
                mse_pp = ((o - ref[None]) ** 2).sum(-1).mean(0)
                bias_raw = ((o.mean(0) - ref) ** 2).sum(-1).mean().item()
                bias2 = max(bias_raw - var_pp.mean().item() / args.R - META[ti]["refvar"], 0.0)
                RES.setdefault((K, v.name), {})[ti] = dict(
                    var=var_pp.cpu(), mse=mse_pp.cpu(), bias2=bias2, varm=var_pp.mean().item(),
                    ess=torch.stack(esss).mean().item(), ms=ms)

    # ---------------------------------------------------------------- report
    nt = len(ts)
    tis_all = list(range(nt))
    sig = [META[ti]["sigma"] for ti in tis_all]
    lo_e, hi_e = args.bands
    bands = {f"low (σ<{lo_e:g})": [ti for ti in tis_all if sig[ti] < lo_e],
             f"mid ({lo_e:g}≤σ<{hi_e:g})": [ti for ti in tis_all if lo_e <= sig[ti] < hi_e],
             f"high (σ≥{hi_e:g})": [ti for ti in tis_all if sig[ti] >= hi_e]}
    agg = make_agg(RES, IDX, args.n_boot)
    w_train = [META[ti]["lam"] for ti in tis_all]

    def w_eq(tis):
        return [1.0 / META[ti]["ref_norm"] for ti in tis]

    csv_rows = []
    for K in args.Ks:
        vs = VARS[K]
        names = [v.name for v in vs]
        base_of = {v.name: v.base for v in vs}
        units = {v.name: v.units for v in vs}
        group = {v.name: v.group for v in vs}
        W = max(len(n) for n in names) + 1
        bn = "repo_cvsi"
        rep.p("")
        rep.p("=" * 110)
        rep.p(f"RESULTS for K = {K} proposal points per score estimate")
        rep.p("=" * 110)

        # health of the baseline
        rep.p("Baseline health (repo_cvsi): at each noise level, how much of its error is noise vs bias, and the "
              "reference accuracy")
        rep.p(f"  {'sigma':>8s}{'relVar':>10s}{'bias%':>8s}{'ESS':>8s}{'refFloor%':>11s}")
        warn_sig = []
        for ti in tis_all:
            r = RES[(K, bn)][ti]
            bshare = 100 * r["bias2"] / max(r["varm"] + r["bias2"], 1e-30)
            if bshare > 50:
                warn_sig.append(sig[ti])
            rep.p(f"  {sig[ti]:8.3g}{r['varm'] / META[ti]['ref_norm']:10.3g}{bshare:8.1f}{r['ess']:8.1f}"
                  f"{100 * META[ti]['refvar'] / META[ti]['ref_norm']:11.3f}")
        rep.p("  relVar = baseline variance / mean |true score|^2 (size of the noise relative to the signal).")
        if warn_sig:
            rep.p("  NOTE: at sigma = " + ", ".join(f"{s:.3g}" for s in warn_sig) + " more than half of the baseline's "
                  "error is BIAS, which variance reduction cannot remove. There, look at ESS (needs better "
                  "proposals), not at variance ratios.")

        # A. variance ratio by noise level
        rep.p("")
        rep.p(f"A. VARIANCE RATIO BY NOISE LEVEL (K={K})   <1 = less noise than baseline; ✓ real gain, ✗ real loss, ~ unclear")
        rep.p(f"{'variant':<{W}}" + "".join(f"{s:>8.3g}" for s in sig) + "    <- sigma")
        last_g = None
        for n in names:
            if group[n] != last_g:
                rep.p(f"-- {group[n]}")
                last_g = group[n]
            if base_of[n] == n:
                continue
            cells = []
            for ti in tis_all:
                r, lo, hi = agg(K, n, base_of[n], [ti], [1.0], "var")
                cells.append(f"{cell(r, lo, hi):>8s}")
            rep.p(f"{n:<{W}}" + "".join(cells))

        # B. summary
        rep.p("")
        rep.p(f"B. SUMMARY OVER NOISE LEVELS (K={K})   ratios vs the row's baseline; [ ] = 95% bootstrap CI")
        rep.p(f"{'variant':<{W}}{'units':>6s}  {'train-wtd var ratio':>22s} {'MSE ratio':>10s}  "
              + "  ".join(f"{b.split(' ')[0]:>6s}" for b in bands) + f"  {'x cost':>7s}{'ESS':>7s}")
        last_g = None
        scope_rows = {}
        for n in names:
            if group[n] != last_g:
                rep.p(f"-- {group[n]}")
                last_g = group[n]
            if base_of[n] == n:
                continue
            b = base_of[n]
            r, lo, hi = agg(K, n, b, tis_all, w_train, "var")
            rm, lom, him = agg(K, n, b, tis_all, w_train, "mse")
            band_cells = []
            for bname, tis in bands.items():
                if tis:
                    rb_, lob, hib = agg(K, n, b, tis, w_eq(tis), "var")
                    band_cells.append(f"{cell(rb_, lob, hib):>6s}")
                    scope_rows.setdefault(bname, []).append((rb_, lob, hib, n))
                else:
                    band_cells.append(f"{'-':>6s}")
            scope_rows.setdefault("train-weighted (what the loss feels)", []).append((r, lo, hi, n))
            msv = sum(RES[(K, n)][ti]["ms"] for ti in tis_all) / nt
            msb = sum(RES[(K, b)][ti]["ms"] for ti in tis_all) / nt
            ess = sum(RES[(K, n)][ti]["ess"] for ti in tis_all) / nt
            rep.p(f"{n:<{W}}{units[n]:>6d}  {r:7.2f} [{lo:4.2f},{hi:4.2f}] {SYM[verdict(lo, hi)]}  "
                  f"{rm:5.2f}{SYM[verdict(lom, him)]}   " + "  ".join(band_cells)
                  + f"  {r * msv / max(msb, 1e-9):7.2f}{ess:7.1f}")
            csv_rows.append(dict(K=K, scope="train_weighted", variant=n, baseline=b, group=group[n], units=units[n],
                                 var_ratio=r, var_lo=lo, var_hi=hi, mse_ratio=rm, ms=msv, ess=ess))
        rep.p("  columns: 'train-wtd' uses the training loss weights; low/mid/high are equal-weight bands of sigma; "
              "'x cost' = var ratio x time ratio (<1 still wins at equal compute).")

        # C. optional per-level detail
        if args.detail:
            for ti in tis_all:
                rep.p("")
                rep.p(f"DETAIL sigma={sig[ti]:.3g}  (K={K}): variant | relVar | var ratio [CI] | MSE ratio | bias% | ESS | ms")
                for n in names:
                    r_ = RES[(K, n)][ti]
                    vr, vlo, vhi = agg(K, n, base_of[n], [ti], [1.0], "var")
                    mr, _, _ = agg(K, n, base_of[n], [ti], [1.0], "mse")
                    bsh = 100 * r_["bias2"] / max(r_["varm"] + r_["bias2"], 1e-30)
                    rep.p(f"  {n:<{W}}{r_['varm'] / META[ti]['ref_norm']:9.3g}  {vr:5.2f} [{vlo:4.2f},{vhi:4.2f}]"
                          f"{SYM[verdict(vlo, vhi)]}  {mr:5.2f}  {bsh:5.1f}  {r_['ess']:6.1f}  {r_['ms']:6.1f}")
                    csv_rows.append(dict(K=K, scope=f"sigma={sig[ti]:.4g}", variant=n, baseline=base_of[n],
                                         group=group[n], units=units[n], var_ratio=vr, var_lo=vlo, var_hi=vhi,
                                         mse_ratio=mr, ms=r_["ms"], ess=r_["ess"]))

        # D. plain-English findings
        rep.p("")
        rep.p(f"C. PLAIN-ENGLISH FINDINGS (K={K}; variance ratios, 95% CI)")
        for label, rows in scope_rows.items():
            better = sorted(x for x in rows if verdict(x[1], x[2]) == "better")
            worse = [x for x in rows if verdict(x[1], x[2]) == "WORSE"]
            txt = (f"clearly better: " + ", ".join(f"{n} ({r:.2f}x [{lo:.2f},{hi:.2f}])" for r, lo, hi, n in better[:3])
                   if better else "no variant is clearly better than its baseline")
            if worse:
                txt += "; clearly worse: " + ", ".join(f"{n} ({r:.2f}x)" for r, _, _, n in worse[:4])
            rep.p(f"  - {label}: {txt}")
        rep.p("  Reminder: this measures the quality of the score ESTIMATE, not final sample quality. Variants that "
              "win here still need training runs with several seeds.")

        # E. plot
        if not args.no_plot:
            try:
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                rows = sorted(scope_rows.get("train-weighted (what the loss feels)", []))[:6]
                fig, ax = plt.subplots(figsize=(8, 5))
                for _, _, _, n in rows:
                    ys = [agg(K, n, base_of[n], [ti], [1.0], "var")[0] for ti in tis_all]
                    ax.plot(sig, ys, marker="o", label=n)
                ax.axhline(1.0, color="k", ls="--", lw=1)
                ax.set_xscale("log")
                ax.set_xlabel("noise level sigma")
                ax.set_ylabel("variance ratio vs baseline (below 1 = better)")
                ax.set_title(f"Best 6 variants by training-weighted variance ratio (K={K})")
                ax.legend(fontsize=7)
                fig.tight_layout()
                fig.savefig(f"{args.out_prefix}_K{K}.png", dpi=150)
                plt.close(fig)
            except Exception as err:  # plotting must never kill the report
                rep.p(f"(plot skipped: {err!r})")

    with open(args.out_prefix + ".csv", "w", newline="") as f:
        if csv_rows:
            wr = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            wr.writeheader()
            wr.writerows(csv_rows)
    rep.save(args.out_prefix + "_report.txt")
    print(f"\nwrote {args.out_prefix}_report.txt, {args.out_prefix}.csv" + ("" if args.no_plot else ", plots"))


if __name__ == "__main__":
    main()