"""Small deterministic sampling specifications used by the dataset generator."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    """Closed scalar interval used for fixed values or uniform sampling."""

    lower: float
    upper: float

    def __post_init__(self) -> None:
        lower = float(self.lower)
        upper = float(self.upper)
        if lower > upper:
            raise ValueError("ParameterSpec requires lower <= upper")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    @classmethod
    def from_raw(
        cls, value: float | int | list[float] | tuple[float, float]
    ) -> "ParameterSpec":
        """Build a specification from one fixed value or two interval bounds."""
        if isinstance(value, (int, float)):
            return cls(lower=float(value), upper=float(value))
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return cls(lower=float(value[0]), upper=float(value[1]))
        raise ValueError("ParameterSpec expects a scalar or two-element bounds")
