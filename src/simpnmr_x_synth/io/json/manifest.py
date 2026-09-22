"""Write dataset provenance manifests."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from simpnmr_x.__version__ import __version__ as simpnmr_x_version

from simpnmr_x_synth.__version__ import __version__

if TYPE_CHECKING:
    from simpnmr_x_synth.cfg.dataset import DatasetGenerationConfig


def write_manifest(
    *, config: DatasetGenerationConfig, output_file: Path, geometry_checksum: str
) -> None:
    """Write the full provenance contract for a generated dataset."""
    payload = {
        "schema_version": 1,
        "generator": {
            "name": "SimpNMR-X-Synth",
            "version": __version__,
            "simpnmr_x_version": simpnmr_x_version,
        },
        "project_name": config.project.name,
        "seed": config.project.seed,
        "n_cases": config.project.n_cases,
        "number_of_moments": config.number_of_moments,
        "geometry_checksum": geometry_checksum,
        "normalized_config": asdict(config),
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
