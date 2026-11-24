from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def sample_iid_2d(k: int, rng: np.random.Generator) -> np.ndarray:
  """Standard i.i.d. N(0, I_2) samples."""
  return rng.standard_normal((k, 2))


def sample_rotstrat_2d(
    n_strata: int, n_rot: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
  """
    Rotational stratification ('rotstrat') in 2D for K = n_strata * n_rot draws.
    - Stratifies the radial quantile u in [0, 1) across n_strata equal bins.
    - Converts u_a -> r_a = sqrt(-2 * ln(1 - u_a)) so marginal r has N(0, I) radial law.
    - Places n_rot equally spaced angles around each circle with a random phase offset.
    """
  # 1. Stratified uniform quantiles -> exact N(0, I_2) radii per stratum
  u_bins = (np.arange(n_strata) + rng.random(n_strata)) / n_strata
  radii = np.sqrt(-2.0 * np.log(1.0 - u_bins))  # shape: (n_strata,)

  # 2. Equally spaced angles + random phase per stratum (Rao-Blackwellized angle)
  phases = rng.uniform(0.0, 2.0 * np.pi, size=(n_strata, 1))
  angles = phases + (2.0 * np.pi * np.arange(n_rot)[None, :]) / n_rot

  # 3. Cartesian coordinates: shape (n_strata, n_rot, 2)
  x = radii[:, None] * np.cos(angles)
  y = radii[:, None] * np.sin(angles)
  points = np.stack([x, y], axis=-1)
  return points, radii


def save_comparison(
    n_strata: int = 12,
    n_rot: int = 8,
    seed: int = 42,
    output_path: str | Path = "rotstrat_comparison.png",
    dpi: int = 300,
) -> Path:
  rng = np.random.default_rng(seed)
  k_total = n_strata * n_rot

  pts_iid = sample_iid_2d(k_total, rng)
  pts_rotstrat, radii = sample_rotstrat_2d(n_strata, n_rot, rng)
  pts_flat = pts_rotstrat.reshape(-1, 2)

  fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharex=True, sharey=True)

  # Left: i.i.d. sampling
  ax = axes[0]
  ax.scatter(
      pts_iid[:, 0], pts_iid[:, 1], s=28, alpha=0.8, edgecolors="none"
  )
  ax.set_title(f"i.i.d. Normal Draws (K = {k_total})")
  ax.set_xlabel("$z_1$")
  ax.set_ylabel("$z_2$")
  ax.set_aspect("equal")
  ax.grid(True, linestyle=":", alpha=0.5)

  # Right: rotstrat sampling
  ax = axes[1]
  theta_grid = np.linspace(0, 2 * np.pi, 200)
  for r in radii:
    ax.plot(
        r * np.cos(theta_grid),
        r * np.sin(theta_grid),
        linestyle="--",
        linewidth=0.7,
        color="gray",
        alpha=0.45,
    )

  cmap = plt.get_cmap("tab20")
  for a in range(n_strata):
    grp = pts_rotstrat[a]  # (n_rot, 2)
    color = cmap(a % 20)
    closed_loop = np.vstack([grp, grp[0]])
    ax.plot(
        closed_loop[:, 0],
        closed_loop[:, 1],
        linewidth=0.8,
        alpha=0.55,
        color=color,
    )
    ax.scatter(
        grp[:, 0],
        grp[:, 1],
        s=32,
        color=color,
        edgecolors="black",
        linewidths=0.4,
    )

  ax.set_title(
      f"rotstrat ({n_strata} radial strata × {n_rot} angles, K = {k_total})"
  )
  ax.set_xlabel("$z_1$")
  ax.set_aspect("equal")
  ax.grid(True, linestyle=":", alpha=0.5)

  lim = max(3.5, np.max(np.abs(pts_flat)) * 1.1)
  ax.set_xlim(-lim, lim)
  ax.set_ylim(-lim, lim)

  out = Path(output_path)
  fig.tight_layout()
  fig.savefig(out, dpi=dpi, bbox_inches="tight")
  plt.close(fig)
  return out


if __name__ == "__main__":
  saved_file = save_comparison(
      n_strata=12, n_rot=8, seed=42, output_path="rotstrat_comparison.png"
  )
  print(f"Saved figure to {saved_file.resolve()}")