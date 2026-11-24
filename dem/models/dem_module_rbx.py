"""DEMLitModule with the extended CVSI estimator (stratified rotation + defensive mixture + multi-control)
and MODE-COVERAGE logging for GMM-40.  dem_module.py is NOT modified.

Select with  model._target_=dem.models.dem_module_rbx.DEMLitModuleRBX  (see configs/experiment/gmm_idem_rbx.yaml).

Mode coverage (how many of the 40 Gaussian modes the generated samples reach):
  * every sample is assigned to its nearest mode; it only counts if it is within `cov_radius_sigmas` std of that mode
  * a mode is "covered" if it got at least max(1, cov_min_frac * N / 40) samples  (default: 10% of its fair share)
Logged every epoch on the model's freshly generated samples (gen/modes/*) and once at test time (test/modes/*),
and appended to  <run dir>/mode_coverage.csv .
"""
import csv
import math
import os
from functools import partial
from typing import Optional

import torch

from .components.cvsi_rbx import idem_score_cvsi_rbx_fn
from .dem_module import DEMLitModule


def mode_coverage(x_real, locs, sd, radius_sigmas=3.0, min_frac=0.1):
    """x_real (N,D) samples in REAL units; locs (M,D) mode centres; sd (M,) mode std. Returns a dict of floats."""
    N, M = x_real.shape[0], locs.shape[0]
    d_min, idx = torch.cdist(x_real, locs).min(dim=1)
    near = d_min <= radius_sigmas * sd[idx]
    counts = torch.bincount(idx[near], minlength=M).float()
    thresh = max(1.0, min_frac * N / M)
    p = counts / counts.sum().clamp_min(1.0)
    ent = -(p[p > 0] * p[p > 0].log()).sum() / math.log(M)  # 1 = perfectly balanced over all modes
    return dict(
        covered=float((counts >= thresh).sum()),
        covered_any=float((counts >= 1).sum()),
        near_frac=float(near.float().mean()),
        balance=float(ent),
    )


def get_gmm_modes(energy_function, device, dtype):
    gmm = getattr(energy_function, "gmm", None)
    locs = getattr(gmm, "locs", None)
    if locs is None:
        raise AttributeError("could not find the GMM mode centres (energy_function.gmm.locs); tell me what attributes exist")
    locs = locs.to(device=device, dtype=dtype)
    st = getattr(gmm, "scale_trils", None)
    if st is not None:
        sd = torch.diagonal(st, dim1=-2, dim2=-1).mean(-1).to(device=device, dtype=dtype)
    else:
        sd = torch.full((locs.shape[0],), 1.313, device=device, dtype=dtype)
    return locs, sd


class DEMLitModuleRBX(DEMLitModule):
    def __init__(
        self,
        *args,
        rb_mode: str = "rotstrat",
        rb_nq: int = 10,
        def_alpha: float = 0.0,
        def_scale: float = 0.6,
        def_center: float = 0.0,
        def_min_sigma: float = 0.3,
        multi_control: bool = False,
        mc_min_ess: float = 100.0,
        mc_ridge: float = 1e-6,
        cov_radius_sigmas: float = 3.0,
        cov_min_frac: float = 0.1,
        **kwargs,
    ) -> None:
        kwargs["use_cvsi"] = True
        super().__init__(*args, **kwargs)
        if rb_mode not in ("none", "rot", "rotstrat"):
            raise ValueError(f"rb_mode must be none|rot|rotstrat, got {rb_mode!r}")
        if self.num_estimator_mc_samples % int(rb_nq) != 0:
            raise ValueError(f"num_estimator_mc_samples must be divisible by rb_nq={rb_nq}")
        self.rbx_kwargs = dict(
            rb_mode=rb_mode, rb_nq=int(rb_nq),
            def_alpha=float(def_alpha), def_scale=float(def_scale), def_center=float(def_center),
            def_min_sigma=float(def_min_sigma),
            multi_control=bool(multi_control), mc_min_ess=float(mc_min_ess), mc_ridge=float(mc_ridge),
        )
        self.cov_radius_sigmas = float(cov_radius_sigmas)
        self.cov_min_frac = float(cov_min_frac)

        grad_fxn = partial(idem_score_cvsi_rbx_fn, debug=False, clipper_tsi=self.clipper_tsi, **self.rbx_kwargs)
        if self.clipper is not None:
            self.clipped_grad_fxn = self.clipper.wrap_grad_fxn(grad_fxn)
        else:
            self.clipped_grad_fxn = grad_fxn

    # ------------------------------------------------------------------ loss
    def get_loss(self, times: torch.Tensor, samples: torch.Tensor, prefix: str = "train") -> torch.Tensor:
        stats = {}
        grad_fxn = partial(idem_score_cvsi_rbx_fn, debug=True, clipper_tsi=self.clipper_tsi, stats=stats, **self.rbx_kwargs)
        (estimated_score, w, tilde_ct, tsi_broad, dsi_broad, cvsi_broad, log_w) = grad_fxn(
            times, samples, self.energy_function, self.noise_schedule, num_mc_samples=self.num_estimator_mc_samples,
        )
        self._log_cvsi_diagnostics(times, tilde_ct, tsi_broad, prefix)
        if getattr(self, "_trainer", None) is not None:
            with torch.no_grad():
                ess = (1.0 / (w.squeeze(-1) ** 2).sum(0)).mean()
                self.log(f"{prefix}/rb/ess", ess, on_step=prefix == "train", on_epoch=True)
                for k, v in stats.items():
                    self.log(f"{prefix}/rbx/{k}", v, on_step=prefix == "train", on_epoch=True)

        # ---- identical to DEMLitModule.get_loss from here ----
        if self.clipper is not None and self.clipper.should_clip_scores:
            if self.energy_function.is_molecule:
                estimated_score = estimated_score.reshape(
                    -1, self.energy_function.n_particles, self.energy_function.n_spatial_dim)
            estimated_score = self.clipper.clip_scores(estimated_score)
            if self.energy_function.is_molecule:
                estimated_score = estimated_score.reshape(-1, self.energy_function.dimensionality)
        if self.score_scaler is not None:
            estimated_score = self.score_scaler.scale_target_score(estimated_score, times)
        predicted_score = self.forward(times, samples)
        error_norms = (predicted_score - estimated_score).pow(2).mean(-1)
        return self.lambda_weighter(times) * error_norms

    # ------------------------------------------------------------------ mode coverage
    @torch.no_grad()
    def _log_mode_coverage(self, samples_norm, prefix):
        x = self.energy_function.unnormalize(samples_norm)  # back to real units, where the 40 modes live
        locs, sd = get_gmm_modes(self.energy_function, x.device, x.dtype)
        res = mode_coverage(x, locs, sd, self.cov_radius_sigmas, self.cov_min_frac)
        for k, v in res.items():
            self.log(f"{prefix}/modes/{k}", v, on_step=False, on_epoch=True)
        try:
            root = getattr(self.trainer, "default_root_dir", ".") or "."
            path = os.path.join(root, "mode_coverage.csv")
            new = not os.path.exists(path)
            with open(path, "a", newline="") as f:
                wr = csv.writer(f)
                if new:
                    wr.writerow(["prefix", "epoch"] + list(res.keys()))
                wr.writerow([prefix, int(self.current_epoch)] + [res[k] for k in res])
        except Exception:
            pass

    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()
        self._log_mode_coverage(self.last_samples, "gen")

    def on_test_epoch_end(self) -> None:
        super().on_test_epoch_end()
        with torch.no_grad():
            samples = self.generate_samples(
                num_samples=self.eval_batch_size, diffusion_scale=self.diffusion_scale,
                negative_time=self.negative_time,
            )
        self._log_mode_coverage(samples, "test")