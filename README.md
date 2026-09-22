# SimpNMR-X-Synth

`SimpNMR-X-Synth` is a deterministic synthetic-data generator for
paramagnetic NMR (pNMR) workflows. It creates replayable datasets for
supervised learning, regression testing, and validation of
[SimpNMR-X](https://github.com/Mephistos-ML/simpnmr-x) fitting pipelines.

It is designed as a companion tool:
generated cases follow the SimpNMR-X example layout, preserve the same data
contracts, and include the information required to reproduce each case.

SimpNMR-X-Synth is part of the broader
[SimpNMR](https://simpnmr.org/) ecosystem.

## Why use SimpNMR-X-Synth?

- Generate deterministic datasets from a single YAML specification.
- Produce paired NMR observables, physical targets, and replayable fit inputs.
- Validate susceptibility and linewidth recovery against known truth.
- Create structured datasets for supervised-learning experiments.
- Record configuration, units, provenance, and generation metadata.

The generator is intended for research software development where reproducible
inputs and explicit validation are as important as the numerical result.

## Package identity

| Interface | Value |
| --- | --- |
| PyPI package | `simpnmr-x-synth` |
| CLI command | `simpnmr-x-synth` |
| Python import | `simpnmr_x_synth` |
| Companion package | `simpnmr-x` |

## Installation

Install the published package and its SimpNMR-X dependency with:

```bash
python3 -m pip install simpnmr-x-synth
```

For development from a local checkout:

```bash
python3 -m pip install -e ".[dev]"
```

## Quick start

Generate a dataset from a YAML configuration, replay one generated case through
SimpNMR-X, and validate the complete dataset:

```bash
simpnmr-x-synth dataset generate ybl8.yml --output datasets/yb_v1

cd datasets/yb_v1/cases/<sample_id>/SIMULATIONS/FITTING
MPLBACKEND=Agg simpnmr-x --hide fit_susc config.yml

cd ../../../..
simpnmr-x-synth dataset validate .
```

The validation command writes `validation_report.json` with generated truth,
fitted values, and errors. A sample is not silently rejected based only on
rank, condition number, or score.

## Dataset layout

Each generated dataset contains a manifest, one canonical machine-learning
table, and one self-contained SimpNMR-X case per sample:

```text
dataset.csv
manifest.json
cases/<sample_id>/
  DATA/
    PARA/generated_shifts.csv
    HFC/geometry.xyz
    DIA/diamagnetic.csv
    CHI/susceptibility.csv
    LABELS/labels.csv              # optional
  SIMULATIONS/
    FITTING/config.yml
```

`dataset.csv` is the canonical supervised-learning artifact. Each row contains
`m1..mN` moment features, six Cartesian susceptibility components, and `p1,p2`
linewidth targets. The selected susceptibility unit is recorded in
`manifest.json`.

The generated case files remain compatible with the SimpNMR-X workflow:

- `DATA/PARA/generated_shifts.csv` contains synthetic paramagnetic shifts.
- `DATA/HFC/geometry.xyz` contains the molecular geometry used for generation.
- `DATA/DIA/diamagnetic.csv` is atom-resolved and normalized to
  `atom_label,shift`.
- `DATA/CHI/susceptibility.csv` contains susceptibility truth.
- `DATA/LABELS/labels.csv` is included when chemical labels are configured.
- `SIMULATIONS/FITTING/config.yml` is a replayable fitting configuration.

Linewidth truth remains exclusively in the root `dataset.csv`.

## Dataset configuration

The following configuration generates a deterministic YbL8 moment dataset:

```yaml
project:
  name: ybl8_moments_v1
  n_cases: 1000
  seed: 42
hyperfine:
  method: pdip
  file: geometries/YbL8.xyz
  paramagnetic_centre: [0.0, 0.0, 0.0]
  spin: 0.5
  orbit: 3
  total_momentum_J: 3.5
nuclei:
  include: H
diamagnetic:
  method: csv
  file: inputs/diamagnetic.csv
signal_labels:                 # optional
  file: inputs/labels.csv
experiment:
  temperature_k: 302.15
  magnetic_field_t: 4.7
moments:
  number_of_moments: 10
linewidth:
  method: r6
susceptibility:
  model: isoaxrho_euler
  rho_over_ax: 0.0             # optional fixed tensor orientation
  alpha: 0.0
  beta: 0.0
  gamma: 0.0
```

When `signal_labels.file` is supplied, the generator creates one peak per
chemical label using SimpNMR-X's `average_shifts: all` policy. Exported replay
and GMM configurations contain the same label mapping.

### Susceptibility models

`susceptibility.model` supports two generation bases:

- `isoaxrho_euler` samples axial/rhombic invariants and ZYZ Euler angles.
- `split` samples the traceless Cartesian components
  `dxx`, `dyy`, `dxy`, `dxz`, and `dyz`.

Both modes derive `chi_iso` from the SimpNMR-X spin-only Curie-law model and
write the same Cartesian tensor targets. A `split` configuration may optionally
fix all five Cartesian deviation components:

```yaml
susceptibility:
  model: split
  dxx: 0.012
  dyy: -0.008
  dxy: 0.003
  dxz: -0.002
  dyz: 0.005
```

Either provide all five split components or omit all of them for deterministic
sampling. `rho_over_ax` and Euler-angle fields apply only to `isoaxrho_euler`.

## Generation model

SimpNMR-X-Synth uses the scientific implementations and conventions provided by
SimpNMR-X:

- `chi_iso` is calculated using the spin-only Curie-law implementation.
- For `isoaxrho_euler`, explicit rhombicity ratios and Euler angles are fixed;
  omitted values are sampled deterministically.
- `rho_over_ax` is sampled in `[0, 1/3]` and Euler angles in standard ZYZ
  domains when not fixed.
- For `split`, Cartesian deviations are sampled deterministically and scaled
  so that the generated susceptibility tensor has non-negative eigenvalues.
- Susceptibility targets are exported in canonical SimpNMR-X units of Å³.
- For `linewidth.method: r6`, `p1` is derived from the point-dipole Guéron
  Curie R2 calculation with the fixed generation policy `tau_R = 1 ns`.
- The distance-independent `p2` is sampled uniformly in `[0, 50] Hz` and
  converted to the ppm convention required by the SimpNMR-X R6 model.

Neither linewidth coefficient is a user-facing configuration parameter.

## Validation stages

The replay profile uses SimpNMR-X fixed assignment to validate the forward data
contract and separately recover R6 `p1,p2` from labelled linewidths.

Assignment-free GMM moment validation belongs to SimpNMR-X's synthetic test
suite and is intentionally treated as a separate validation stage.

## Development

Run the unit and integration suites with:

```bash
python3 -m pytest -m 'not integration'
python3 -m pytest -m integration
```

The integration suite launches the real `simpnmr-x` executable and must run
against a compatible SimpNMR-X version. Every generated CSV records the
`SimpNMR-X-Synth` version in its comment header.

## Links

- [SimpNMR-X](https://github.com/Mephistos-ML/simpnmr-x)
- [SimpNMR-X documentation](https://mephistos-ml.github.io/simpnmr-x/)
- [SimpNMR-X-Synth on PyPI](https://pypi.org/project/simpnmr-x-synth/)
- [SimpNMR](https://simpnmr.org/)
- [MIT License](LICENSE)
