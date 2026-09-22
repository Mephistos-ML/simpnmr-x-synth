"""Typed records for synthetic moment-matching datasets."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TensorTarget:
    """Cartesian χ tensor and global R6 linewidth targets."""

    chi_xx: float
    chi_xy: float
    chi_xz: float
    chi_yy: float
    chi_yz: float
    chi_zz: float
    linewidth_p1: float
    linewidth_p2: float

    def __post_init__(self) -> None:
        for field_name in self.__dataclass_fields__:
            object.__setattr__(self, field_name, float(getattr(self, field_name)))

    def as_row(self) -> dict[str, float]:
        """Return the canonical target-column mapping."""
        return {
            "chi_xx": self.chi_xx,
            "chi_xy": self.chi_xy,
            "chi_xz": self.chi_xz,
            "chi_yy": self.chi_yy,
            "chi_yz": self.chi_yz,
            "chi_zz": self.chi_zz,
            "linewidth_p1": self.linewidth_p1,
            "linewidth_p2": self.linewidth_p2,
        }


@dataclass(frozen=True, slots=True)
class DatasetRecord:
    """One ML-facing moment vector and its tensor target.

    Args:
        sample_id: Globally unique deterministic sample identifier.
        temperature_k: Temperature for this tensor point in K.
        magnetic_field_t: Magnetic field in T.
        moments: Ordered mapping of dynamic ``m1`` through ``mN`` values.
        target: Cartesian χ tensor and R6 linewidth target.
    """

    sample_id: str
    temperature_k: float
    magnetic_field_t: float
    moments: dict[str, float]
    target: TensorTarget

    def __post_init__(self) -> None:
        if not self.sample_id:
            raise ValueError("sample_id must be non-empty")
        if self.temperature_k <= 0.0 or self.magnetic_field_t <= 0.0:
            raise ValueError("temperature and magnetic field must be positive")
        if not self.moments:
            raise ValueError("DatasetRecord requires at least one moment")
        names = tuple(self.moments)
        expected = tuple(f"m{index}" for index in range(1, len(names) + 1))
        if names != expected:
            raise ValueError("moments must be ordered consecutively from m1")
        object.__setattr__(
            self,
            "moments",
            {name: float(value) for name, value in self.moments.items()},
        )
