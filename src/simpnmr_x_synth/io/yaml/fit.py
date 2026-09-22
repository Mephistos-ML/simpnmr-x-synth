"""Write runnable SimpNMR-X replay and assignment-free GMM configurations."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from simpnmr_x_synth.cfg.dataset import DatasetGenerationConfig


def write_fit_config(*, config: DatasetGenerationConfig, output_file: Path) -> None:
    """Write the self-contained SimpNMR-X replay configuration."""
    payload = {
        "project": {"name": "simpnmr_x_fitted_output"},
        "hyperfine": {
            "method": "pdip", "file": "../../DATA/HFC/geometry.xyz",
            "paramagnetic_centre": list(config.hyperfine.paramagnetic_centre),
            "spin": config.hyperfine.spin, "orbit": config.hyperfine.orbit,
            "total_momentum_J": config.hyperfine.total_momentum_j,
        },
        "nuclei": {"include": config.nuclei_include},
        "diamagnetic": {
            "method": "csv",
            "file": "../../DATA/DIA/diamagnetic.csv",
        },
        "experiment": {"files": "../../DATA/PARA/generated_shifts.csv"},
        "assignment": {"method": "fixed"},
        "linewidth": {"method": "experimental", "estimate": "p1_p2"},
        "susc_fit": {
            "type": "isoaxrho_euler",
            "variables": {
                "iso": ["fit", 0.0], "ax": ["fit", 0.01], "rho_over_ax": ["fit", 0.1],
                "alpha": ["fit", 0.0], "beta": ["fit", 0.0], "gamma": ["fit", 0.0],
            },
        },
    }
    if config.signal_labels_file:
        payload["signal_labels"] = {"file": "../../DATA/LABELS/labels.csv"}
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def write_gmm_fit_config(*, config: DatasetGenerationConfig, output_file: Path) -> None:
    """Write an assignment-free GMM fit in Cartesian split coordinates."""
    payload = {
        "project": {"name": "simpnmr_x_gmm_fitted_output"},
        "hyperfine": {
            "method": "pdip", "file": "../../DATA/HFC/geometry.xyz",
            "paramagnetic_centre": list(config.hyperfine.paramagnetic_centre),
            "spin": config.hyperfine.spin, "orbit": config.hyperfine.orbit,
            "total_momentum_J": config.hyperfine.total_momentum_j,
        },
        "nuclei": {"include": config.nuclei_include},
        "diamagnetic": {"method": "csv", "file": "../../DATA/DIA/diamagnetic.csv"},
        "experiment": {"files": "../../DATA/PARA/generated_shifts.csv"},
        "assignment": {
            "method": "moments",
            "max_moment_order": config.number_of_moments,
        },
        "linewidth": {
            "method": "r6",
            "variables": {
                "p1": ["fit", 1.0, [0.0, 1.0e6]],
                "p2": ["fit", 0.1, [0.0, 10.0]],
            },
        },
        "susc_fit": {
            "type": "split",
            "variables": {
                "iso": ["fix", 0.0],
                "dxx": ["fit", 0.0],
                "dyy": ["fit", 0.0],
                "dxy": ["fit", 0.0],
                "dxz": ["fit", 0.0],
                "dyz": ["fit", 0.0],
            },
        },
    }
    if config.signal_labels_file:
        payload["signal_labels"] = {"file": "../../DATA/LABELS/labels.csv"}
        payload["susc_fit"]["average_shifts"] = "all"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
