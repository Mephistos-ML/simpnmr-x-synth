# ParaNMR-Synth — AI Agent Development Contract

## 1. Scope and Authority

This contract is mandatory for every AI agent modifying this repository. It
overrides an agent's default implementation preferences. Non-compliant changes
must not be proposed or applied.

## 2. Product Boundary

`ParaNMR-Synth` is the reproducible synthetic-dataset orchestrator for the
ParaNMR ecosystem. Its production artifact is a provenance-complete dataset,
not merely a susceptibility-tensor CSV.

ParaNMR is the sole source of truth for scientific pNMR behaviour:

- susceptibility tensor parameterizations and conventions;
- PDA and other hyperfine forward models;
- linewidth calculations;
- Gaussian peak representation and moment descriptors;
- ParaNMR experiment-file formats.

`ParaNMR-Synth` may sample physical latent variables, apply explicitly
configured measurement noise, and write dataset manifests. Dataset splitting
is an explicit downstream ML concern, not an implicit generator policy. It
must call ParaNMR APIs for every scientific calculation.

## 3. Layers and Allowed Dependencies

```text
CLI -> app -> core
             -> io
```

External dependency direction is:

```text
ParaNMR-Synth -> ParaNMR
```

ParaNMR must never import `paranmr_synth`.

- `cli`: argument parsing and dispatch only.
- `cfg`: YAML parsing and validation only.
- `app`: workflow orchestration, dataset assembly, and output coordination.
- `core`: deterministic sampling and noise transforms.
- `io`: serialization and deserialization only.

## 4. Scientific Source-of-Truth Rules

The following are forbidden in `ParaNMR-Synth`:

- reimplementing χ tensor construction, Euler rotations, PDA, PCS, linewidth,
  Gaussian peak, or moment equations;
- calling the `paranmr` CLI or parsing ParaNMR output text to obtain a numeric
  result when a Python API exists or can be added to ParaNMR;
- silently approximating missing ParaNMR functionality.

`ParaNMR-Synth` must never modify ParaNMR. If an API is unavailable, use a
supported existing ParaNMR interface or leave the capability out of scope.

## 5. Dataset Contract

Every dataset must have a manifest recording at least:

- dataset schema version;
- `ParaNMR-Synth` and ParaNMR versions;
- full normalized YAML configuration;
- all random seeds;
- source geometry and input-file checksums;
- sample identifiers;
- requested moment labels.

The number of moments is always configuration-driven. Code must accept every
positive `moments.number_of_moments` value and derive ordered labels from the
canonical `m1` through `mN` convention. Do not hard-code `m1` through `m6`.

Dataset targets must include the six independent Cartesian χ components.
Latent iso/ax/rho/Euler values are auxiliary provenance labels, not the sole
canonical target.

## 6. Reproducibility and Leakage

- Randomness is forbidden unless its seed is explicit in YAML and persisted.
- Noise is applied to peaks before moments are calculated; never add independent
  noise directly to moments.
- Dataset splitting is explicit downstream ML policy, not generator behaviour.
- File ordering, sample identifiers, CSV columns, and floating-point formatting
  must be deterministic.

## 7. Change Discipline

- YAML schemas, dataset columns, units, and metadata are public contracts.
- Do not change a public contract silently.
- Add focused tests for every new behaviour.
- Public functions and classes require type hints and Google-style docstrings.
- Do not add hidden defaults, environment-dependent paths, or debug output.
