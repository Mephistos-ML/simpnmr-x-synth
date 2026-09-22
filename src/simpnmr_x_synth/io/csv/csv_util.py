"""CSV serialization primitives for SimpNMR-X-Synth artifacts."""

from __future__ import annotations

import datetime
from os import PathLike
from pathlib import Path

import pandas as pd

from simpnmr_x_synth.__version__ import __version__


def write_csv_safe(
    dataframe: pd.DataFrame,
    file_name: str | PathLike[str],
    comment: str | list[str] | None = None,
    *,
    sep: str = ",",
    float_format: str = "%.15g",
    index: bool = False,
    encoding: str = "utf-8-sig",
    newline: str = "",
) -> None:
    """Write a CSV with SimpNMR-X-Synth provenance and stable UTF-8 encoding.

    This is the synthetic-product serialization primitive. SimpNMR-X remains the
    source of truth for scientific calculations and its own file formats.
    """
    path = Path(file_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding=encoding, newline=newline) as handle:
        timestamp = datetime.datetime.now().strftime("%H:%M:%S %d-%m-%Y")
        handle.write(
            f"# This file was generated with SimpNMR-X-Synth v{__version__} at {timestamp}\n"
        )
        if comment is not None:
            comments = [comment] if isinstance(comment, str) else comment
            for line in comments:
                handle.write(f"# {line}\n")
        dataframe.to_csv(handle, sep=sep, index=index, float_format=float_format)
