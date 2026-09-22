"""Write synthetic susceptibility truth."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from simpnmr_x_synth.core.generators.susceptibility import (
    GeneratedSusceptibility,
    IsoAxRhoEulerLatents,
    SplitLatents,
)
from simpnmr_x_synth.io.csv.csv_util import write_csv_safe

if TYPE_CHECKING:
    from simpnmr_x_synth.core.dataset.records import TensorTarget


def write_susceptibility(
    *, target: TensorTarget, latent: GeneratedSusceptibility, output_file: Path
) -> None:
    """Write Cartesian χ and its sampled parameterization."""
    row = {
        "model": latent.model,
        "chi_xx": target.chi_xx,
        "chi_xy": target.chi_xy,
        "chi_xz": target.chi_xz,
        "chi_yy": target.chi_yy,
        "chi_yz": target.chi_yz,
        "chi_zz": target.chi_zz,
    }
    if isinstance(latent, IsoAxRhoEulerLatents):
        row.update(
            {
                "chi_iso": latent.iso,
                "chi_ax": latent.ax,
                "chi_rh": latent.ax * latent.rho_over_ax,
                "rh_over_ax": latent.rho_over_ax,
                "alpha": latent.alpha,
                "beta": latent.beta,
                "gamma": latent.gamma,
            }
        )
    elif isinstance(latent, SplitLatents):
        decomposition = latent.decomposition
        rho_over_ax = (
            0.0
            if decomposition.axiality == 0.0
            else decomposition.rhombicity / decomposition.axiality
        )
        row.update(
            {
                "chi_iso": decomposition.iso,
                "chi_ax": decomposition.axiality,
                "chi_rh": decomposition.rhombicity,
                "rh_over_ax": rho_over_ax,
                "alpha": decomposition.alpha,
                "beta": decomposition.beta,
                "gamma": decomposition.gamma,
                "dxx": latent.dxx,
                "dyy": latent.dyy,
                "dxy": latent.dxy,
                "dxz": latent.dxz,
                "dyz": latent.dyz,
            }
        )
    else:
        raise TypeError(f"unsupported susceptibility latent type: {type(latent)!r}")
    write_csv_safe(pd.DataFrame([row]), output_file, float_format="%.15g")
