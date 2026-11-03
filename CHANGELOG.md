## 2.0.11 — 2026-11-03

- Correct release and archival descriptions and rebuild distribution metadata.
- Preserve computational code and numerical evidence from version 2.0.10; update the software version identifier.

## 2.0.10

- Publication preflight: canonicalize document line endings in release checks, retain byte-exact module/evidence comparisons, enforce LF text checkouts, and run the Python 3.9 Linux job on Ubuntu 22.04. Add three platform-newline regressions; computational modules remain unchanged.

- 3 November 2026 public release preparation: register the public GitHub repository and complete repository URLs and citation metadata. Computational modules remain unchanged.

- 3 November 2026 release preparation: complete author, affiliation, ORCID and support metadata; preserve upstream license notice and credit.
- Regenerate 2.0.10 numerical evidence and figures and validate rebuilt wheels. Computational modules remain unchanged.

- Parse complete citation YAML safely, reject duplicate keys, and strictly validate complete release version and date fields.
- Cover malformed quotes, date suffixes, duplicate fields, wrong shapes and valid quoted/unquoted dates.
- Install the existing PyYAML development dependency explicitly in CI.

## 2.0.9

- Synchronize CITATION.cff version and preparation date; validate source and archived citation.
- Include and verify the complete frozen validation snapshot in the sdist, including Markdown scope notes.
- Run release verification in CI using distributions/ consistently.
- Separate post-build validation logs from bundled historical and source-test evidence.

## 2.0.8

- Correct current README title and unify platform-validation statements.
- Reject stale or divergent wheel/sdist descriptions and sdist README content.
- Add manual CI execution and retained runtime/log evidence for each platform.
- Windows/macOS execution remains pending; no cross-platform pass is claimed.

# 2.0.7

- Prevent subnormal reliability products from corrupting normalized expert weights.
- Use exponent scaling with exact rational fallback for very small active products.
- Separate raw, scaled and underflow diagnostics in reliability output.
- Add 17 regression cases including Decimal references and full-pipeline scale equivalence.

# 2.0.6

- Repair reliability-dimension weighted sums at unit boundaries without relaxing input checks.
- Vectorize consensus normalization while preserving returned residual semantics.
- Reject missing, extra and duplicate distribution modules, including nested paths.
- Add 16 regression cases; refresh numerical reproduction and measured timings.

# 2.0.5

- Apply bounded roundoff and MFV normalization to consensus target and updates.
- Calculate residuals from the exact matrices returned to callers.
- Build in fresh temporary checkouts; reject obsolete or mixed distributions.
- Verify wheel/sdist metadata, source identity and complete release ZIP checksums.
- Add eight regression cases, including the previously rejected expert weights.

# 2.0.4

- Bound internal convex roundoff without relaxing external input validation.
- Return strict JSON-compatible record-mode scan reports.
- Define raw-score gaps and distinct between-tie-group gaps with label mappings.
- Add 23 regression cases for vertices, JSON reports and tied score gaps.

# 2.0.3

- Recursive finite JSON conversion for extended-precision NumPy input.
- Reproducible failed perturbation metadata, including mutated input hashes.
- Opt-in record-and-continue scans with failure counts and partial stability.
- Shared ACO validation before execution, including inactive options.
- 27 regression cases; actual Python 3.9 / NumPy 1.24 lower-bound validation.

# 2.0.2

- Preserve scoring validity, weight source, consensus/network residuals and warnings in sensitivity records.
- Support custom reliability dimensions in leave-one-out analysis; expose independent components, explicit failed/skipped subsets and trial counts.
- Reject nonboolean keep_log and out-of-range/nonfinite normalized angles.
- Record resolved ACO options, supplied reliability dimensions, ordered input hashes, labels and runtime versions.
- Correct full-reproduction dependency instructions and add regression evidence.

# 2.0.1

- Protect CLI inputs and existing outputs; atomically publish complete JSON; add --force for other outputs.
- Prevent geometry re-registration and remove the registry from raw-angle computation.
- Permit explicit criterion weights when the diagnostic network does not converge.
- Use scaled reciprocal normalization, enforce representable numerical ranges, and reject lossy numeric conversions.
- Support consistent sequence containers and isolate caller-owned configuration.
- Add 37 regression cases (105 total), full installed-wheel checks, dependency-lower-bound CI and measured parameter sensitivity.
- Update manuscript claims and numerical evidence. Default example ranking is unchanged; weight stress tests can change the leading alternative.

# Changes in 2.0.0

Released as a research beta revision on 2026-10-20. This is a breaking model/API change from 1.0.0.

- Default preference ranking is additive MFV utility with explicit benefit/cost direction and tie groups.
- Removed `aco_rank` behavior: it raises an explanatory migration error. Use `aco_sequence` for traversal. Sequence results have `sequence_labels`, not `ranking_labels`.
- Geometry labels remain diagnostic scales; their cancellation is disclosed and evaluated canonically.
- Criterion vectors retain aligned coordinates; zero diagonal is an absent relation.
- Panel-agreement reliability replaces arbitrary odd/even-column stability. User dimensions and explicit weights are accepted by the pipeline.
- LOM is evaluated analytically; the zero-total case is rejected; SciPy is no longer required.
- Consensus uses the reliability-weighted target, convex relaxation, target residual, and cap status. `molfuzzy.qlearning` is a deprecated bridge, not an RL implementation. Benchmark selection is no longer supported for balancing.
- Expert reliability weights now apply to both criterion relations and alternative evaluation matrices.
- Angles use a stable atan2 formula; exact and near duplicates share finite regularization.
- Input shapes, finite numbers, ranges, labels, missing cells, JSON constants and duplicate keys are validated.
- Traversal uses exact enumeration at <=8 alternatives; otherwise minimum-iteration/patience stopping, seed control, canonical order and optional logs.
- Sensitivity distinguishes criterion order, alternative tie groups and traversal; local perturbations and expert deletion are included.
- CLI JSON is machine-readable and exports complete results. Source examples, tests, evidence, figures and packaging scripts are included.
- Production/Stable and unsupported method-validation claims are removed; repository and author facts remain pending rather than invented.

## Migration example

`result.ranking.ranking_labels` now refers to utility preferences. For angular traversal explicitly request `include_sequence=True` and inspect `result.sequence.sequence_labels`. Use `result.ranking.rank_groups` to preserve ties. The former `result.ranking.fitness` becomes `result.sequence.fitness`, and only describes traversal.

The learning rate now controls diagnostic relaxation speed only. A user needing a different substantive consensus model must implement and validate it explicitly; do not select learning rates to produce a desired ranking.
