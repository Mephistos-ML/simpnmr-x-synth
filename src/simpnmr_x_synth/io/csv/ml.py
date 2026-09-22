"""Write paired supervised-learning tables."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from simpnmr_x_synth.io.csv.csv_util import write_csv_safe

if TYPE_CHECKING:
    from simpnmr_x_synth.app.pipelines.dataset_generation import GeneratedCase


def write_ml_dataset(*, cases: tuple[GeneratedCase, ...], output_file: Path) -> None:
    """Write the canonical paired moments-to-parameters ML table."""
    rows = [
        {
            "sample_id": case.record.sample_id,
            **case.record.moments,
            **case.record.target.as_row(),
        }
        for case in cases
    ]
    write_csv_safe(pd.DataFrame(rows), output_file, float_format="%.15g")
