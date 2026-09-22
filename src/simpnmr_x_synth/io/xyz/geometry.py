"""Write indexed geometries for SimpNMR-X replay."""

from __future__ import annotations

from pathlib import Path

from simpnmr_x.io.xyz.xyz_write import save_xyz
from simpnmr_x.tools.coords.xyz_fmt import load_xyz


def write_indexed_geometry(*, input_file: str, output_file: Path) -> None:
    """Write a PDiP-compatible indexed XYZ using SimpNMR-X's XYZ writer."""
    labels, coordinates = load_xyz(input_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    save_xyz(
        str(output_file),
        labels,
        coordinates,
        with_numbers=True,
        verbose=False,
        comment="Indexed by SimpNMR-X-Synth for replayable SimpNMR-X fitting.",
    )
