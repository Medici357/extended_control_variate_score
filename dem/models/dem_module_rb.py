"""DEMLitModule with the Rao-Blackwellised CVSI score estimator (iDEM + CVSI + RB).

Subclass of DEMLitModule: only the score estimator changes (get_loss and clipped_grad_fxn).
dem_module.py, dem.yaml and dem/train.py are NOT modified.

Select it with  model._target_=dem.models.dem_module_rb.DEMLitModuleRB  (see configs/experiment/gmm_idem_rb.yaml).
"""

from functools import partial
from typing import Optional

import torch

from .components.cvsi_rb import idem_score_cvsi_rb_fn
from .dem_module import DEMLitModule


class DEMLitModuleRB(DEMLitModule):
    def __init__(
        self,
        *args,
        rb_mode: str = "coord",
        rb_rule: str = "strat",
        rb_nq: int = 10,
        rb_random_frame: bool = True,
        rb_max_sigma: Optional[float] = None,
        rb_rot_radius: int = 0,
        rb_coef_weighting: str = "prop",
        rb_min_ess: float = 8.0,
        **kwargs,
    ) -> None:
        kwargs["use_cvsi"] = True  # CVSI by construction
        super().__init__(*args, **kwargs)

        if rb_mode not in ("none", "coord", "radial", "rot"):
            raise ValueError(f"rb_mode must be none|coord|radial|rot, got {rb_mode!r}")
        if rb_coef_weighting not in ("prop", "post", "gpost"):
            raise ValueError(f"rb_coef_weighting must be prop|post|gpost, got {rb_coef_weighting!r}")
        if rb_mode == "rot" and int(rb_nq) < 2:
            raise ValueError("rb_mode=rot needs rb_nq >= 2 (rb_nq=2 is antithetic pairs)")
        if rb_mode != "none" and self.num_estimator_mc_samples % int(rb_nq) != 0:
            raise ValueError(
                f"num_estimator_mc_samples={self.num_estimator_mc_samples} must be divisible by rb_nq={rb_nq}"
            )
        self.rb_kwargs = dict(
            rb_mode=rb_mode,
            rb_rule=rb_rule,
            rb_nq=int(rb_nq),
            rb_random_frame=bool(rb_random_frame),
            rb_max_sigma=None if rb_max_sigma is None else float(rb_max_sigma),
            rb_rot_radius=int(rb_rot_radius),
            coef_weighting=rb_coef_weighting,
            min_ess=float(rb_min_ess),
        )

        # Estimator used by the reverse SDE (setup() looks up self.clipped_grad_fxn lazily).
        grad_fxn = partial(
            idem_score_cvsi_rb_fn, debug=False, clipper_tsi=self.clipper_tsi, **self.rb_kwargs
        )
        # parent sets self.clipper only when clip_all=True (same convention as parent).
        if self.clipper is not None:
            self.clipped_grad_fxn = self.clipper.wrap_grad_fxn(grad_fxn)
        else:
            self.clipped_grad_fxn = grad_fxn

    def get_loss(self, times: torch.Tensor, samples: torch.Tensor, prefix: str = "train") -> torch.Tensor:
        grad_fxn = partial(
            idem_score_cvsi_rb_fn, debug=True, clipper_tsi=self.clipper_tsi, **self.rb_kwargs
        )
        (estimated_score, w, tilde_ct, tsi_broad, dsi_broad, cvsi_broad, log_w) = grad_fxn(
            times,
            samples,
            self.energy_function,
            self.noise_schedule,
            num_mc_samples=self.num_estimator_mc_samples,
        )
        self._log_cvsi_diagnostics(times, tilde_ct, tsi_broad, prefix)
        if getattr(self, "_trainer", None) is not None:
            with torch.no_grad():
                ess = (1.0 / (w.squeeze(-1) ** 2).sum(0)).mean()  # weight ESS (nodes on a line are not independent)
                self.log(f"{prefix}/rb/ess", ess, on_step=prefix == "train", on_epoch=True)

        # ---- below: identical to DEMLitModule.get_loss ----
        if self.clipper is not None and self.clipper.should_clip_scores:
            if self.energy_function.is_molecule:
                estimated_score = estimated_score.reshape(
                    -1, self.energy_function.n_particles, self.energy_function.n_spatial_dim
                )
            estimated_score = self.clipper.clip_scores(estimated_score)
            if self.energy_function.is_molecule:
                estimated_score = estimated_score.reshape(-1, self.energy_function.dimensionality)

        if self.score_scaler is not None:
            estimated_score = self.score_scaler.scale_target_score(estimated_score, times)

        predicted_score = self.forward(times, samples)
        error_norms = (predicted_score - estimated_score).pow(2).mean(-1)
        return self.lambda_weighter(times) * error_norms