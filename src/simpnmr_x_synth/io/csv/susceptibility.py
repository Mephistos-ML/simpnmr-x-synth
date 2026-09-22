"""Write synthetic susceptibility truth."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from simpnmr_x_synth.io.csv.csv_util import write_csv_safe

if TYPE_CHECKING:
    from simpnmr_x_synth.core.dataset.records import TensorTarget
    from simpnmr_x_synth.core.generators.susceptibility import SusceptibilityLatents


def write_susceptibility(
    *, target: TensorTarget, latent: SusceptibilityLatents, output_file: Path
) -> None:
    """Write Cartesian χ and its sampled parameterization."""
    row = {
        "chi_xx": target.chi_xx,
        "chi_xy": target.chi_xy,
        "chi_xz": target.chi_xz,
        "chi_yy": target.chi_yy,
        "chi_yz": target.chi_yz,
        "chi_zz": target.chi_zz,
        "chi_iso": latent.iso,
        "chi_ax": latent.ax,
        "chi_rh": latent.ax * latent.rho_over_ax,
        "rh_over_ax": latent.rho_over_ax,
        "alpha": latent.alpha,
        "beta": latent.beta,
        "gamma": latent.gamma,
    }
    write_csv_safe(pd.DataFrame([row]), output_file, float_format="%.15g")
