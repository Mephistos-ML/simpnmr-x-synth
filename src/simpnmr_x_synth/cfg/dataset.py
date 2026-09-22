"""Strict YAML contract for synthetic moment-dataset generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from simpnmr_x_synth.cfg.models import (
    DiamagneticConfig, ExperimentConfig, HyperfineConfig, LinewidthConfig,
    ProjectConfig, SusceptibilityConfig,
)


@dataclass(frozen=True, slots=True)
class DatasetGenerationConfig:
    """Validated public configuration for one synthetic dataset run."""

    project: ProjectConfig
    hyperfine: HyperfineConfig
    signal_labels_file: str
    nuclei_include: str
    diamagnetic: DiamagneticConfig
    experiment: ExperimentConfig
    number_of_moments: int
    linewidth: LinewidthConfig
    susceptibility: SusceptibilityConfig

    @classmethod
    def from_file(cls, file_name: str | Path) -> "DatasetGenerationConfig":
        """Load and validate one dataset-generation YAML file."""
        path = Path(file_name).resolve()
        with path.open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        if not isinstance(raw, dict):
            raise ValueError("Dataset configuration root must be a mapping")
        hyperfine = _mapping(raw, "hyperfine")
        hyperfine_file = Path(str(hyperfine["file"]))
        if not hyperfine_file.is_absolute():
            hyperfine["file"] = str(path.parent / hyperfine_file)
        for section in ("diamagnetic", "diamagnetic_ref", "signal_labels"):
            value = raw.get(section)
            if isinstance(value, dict) and "file" in value:
                file_name = Path(str(value["file"]))
                if not file_name.is_absolute():
                    value["file"] = str(path.parent / file_name)
        return cls.from_mapping(raw)

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "DatasetGenerationConfig":
        """Build the typed contract from parsed YAML content."""
        project = _mapping(raw, "project")
        hyperfine = _mapping(raw, "hyperfine")
        nuclei = _mapping(raw, "nuclei")
        signal_labels = raw.get("signal_labels", {})
        if not isinstance(signal_labels, dict):
            raise ValueError("signal_labels must be a mapping")
        diamagnetic = _mapping(raw, "diamagnetic")
        experiment = _mapping(raw, "experiment")
        moments = _mapping(raw, "moments")
        linewidth = _mapping(raw, "linewidth")
        susceptibility = _mapping(raw, "susceptibility")
        method = str(hyperfine["method"]).lower()
        if method != "pdip":
            raise ValueError("hyperfine.method must be 'pdip'")
        linewidth_method = str(linewidth["method"]).lower()
        if linewidth_method != "r6":
            raise ValueError("linewidth.method must be 'r6'")
        model = str(susceptibility["model"]).lower()
        if model != "isoaxrho_euler":
            raise ValueError("susceptibility.model must be 'isoaxrho_euler'")
        rho_over_ax = _optional_float(susceptibility, "rho_over_ax")
        if rho_over_ax is not None and not 0.0 <= rho_over_ax <= 1.0 / 3.0:
            raise ValueError("susceptibility.rho_over_ax must be in [0, 1/3]")
        centre = tuple(float(value) for value in hyperfine["paramagnetic_centre"])
        if len(centre) != 3:
            raise ValueError("hyperfine.paramagnetic_centre must have three values")
        diamagnetic_method = str(diamagnetic["method"]).lower()
        if diamagnetic_method not in {"csv", "dft"}:
            raise ValueError("diamagnetic.method must be 'csv' or 'dft'")
        diamagnetic_file = _nonempty(diamagnetic["file"], "diamagnetic.file")
        diamagnetic_ref = raw.get("diamagnetic_ref", {})
        if not isinstance(diamagnetic_ref, dict):
            raise ValueError("diamagnetic_ref must be a mapping")
        reference_method = str(diamagnetic_ref.get("method", "")).lower()
        reference_file = str(diamagnetic_ref.get("file", ""))
        if diamagnetic_method == "dft":
            if reference_method != "dft" or not reference_file.strip():
                raise ValueError(
                    "diamagnetic.method 'dft' requires "
                    "diamagnetic_ref.method 'dft' and diamagnetic_ref.file"
                )
        elif reference_method or reference_file.strip():
            if reference_method not in {"csv", "dft"} or not reference_file.strip():
                raise ValueError(
                    "diamagnetic_ref requires both a 'csv' or 'dft' method and file"
                )
        n_cases = int(project["n_cases"])
        number_of_moments = int(moments["number_of_moments"])
        if n_cases <= 0 or number_of_moments <= 0:
            raise ValueError("n_cases and number_of_moments must be positive")
        return cls(
            project=ProjectConfig(_nonempty(project["name"], "project.name"), n_cases, int(project["seed"])),
            hyperfine=HyperfineConfig(_nonempty(hyperfine["file"], "hyperfine.file"), centre, float(hyperfine["spin"]), float(hyperfine["orbit"]), float(hyperfine["total_momentum_J"])),
            signal_labels_file=str(signal_labels.get("file", "")).strip(),
            nuclei_include=_nonempty(nuclei["include"], "nuclei.include"),
            diamagnetic=DiamagneticConfig(
                diamagnetic_method, diamagnetic_file, reference_method, reference_file
            ),
            experiment=ExperimentConfig(float(experiment["temperature_k"]), float(experiment["magnetic_field_t"])),
            number_of_moments=number_of_moments,
            linewidth=LinewidthConfig(linewidth_method),
            susceptibility=SusceptibilityConfig(
                model=model,
                rho_over_ax=rho_over_ax,
                alpha=_optional_float(susceptibility, "alpha"),
                beta=_optional_float(susceptibility, "beta"),
                gamma=_optional_float(susceptibility, "gamma"),
            ),
        )

    @property
    def moment_labels(self) -> tuple[str, ...]:
        """Return dynamic canonical labels from ``m1`` through ``mN``."""
        return tuple(f"m{index}" for index in range(1, self.number_of_moments + 1))


def _mapping(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _nonempty(value: object, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty")
    return text


def _optional_float(raw: dict[str, Any], name: str) -> float | None:
    """Return one optional numeric susceptibility setting."""
    value = raw.get(name)
    return None if value is None else float(value)
