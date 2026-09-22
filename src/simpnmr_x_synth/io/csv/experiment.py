"""Write SimpNMR-X experiment inputs."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from simpnmr_x.core.const.gammas import NUCLEAR_GAMMAS
from simpnmr_x.core.util.strings import remove_numbers

from simpnmr_x_synth.io.csv.csv_util import write_csv_safe

if TYPE_CHECKING:
    from simpnmr_x_synth.app.pipelines.dataset_generation import GeneratedCase
    from simpnmr_x_synth.cfg.dataset import DatasetGenerationConfig


def write_experiment(
    *, config: DatasetGenerationConfig, case: GeneratedCase, output_file: Path
) -> None:
    """Write the generated peak list using SimpNMR-X's experiment writer."""
    gamma = NUCLEAR_GAMMAS[remove_numbers(config.nuclei_include)]
    rows = [
        {
            "signal_label": peak.label,
            "shift (ppm)": peak.center_ppm,
            "width (Hz)": peak.fwhm_ppm * gamma * config.experiment.magnetic_field_t,
            "area ()": peak.area,
            "L/G ()": 1.0,
        }
        for peak in case.peaks
    ]
    dataframe = pd.DataFrame(rows).sort_values("shift (ppm)").reset_index(drop=True)
    write_csv_safe(
        dataframe,
        output_file,
        comment=[
            f"temperature {config.experiment.temperature_k}",
            f"magnetic_field {config.experiment.magnetic_field_t}",
            f"isotope 1{config.nuclei_include}",
        ],
        float_format="%.15g",
    )
