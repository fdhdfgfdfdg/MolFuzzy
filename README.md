# MolFuzzy 2.0.10

A Python research toolkit for explicit fuzzy preference scoring, expert-consensus diagnostics, similarity-network criterion weights, and optional angular traversal analysis. MIT licensed. **Beta research software.**

## Important changes from 1.0.0

This is a breaking scientific and API revision. The default result is an additive utility ranking with explicit benefit/cost directions and tie groups. An ACO visiting sequence is **not** a best-to-worst ranking. The optional traversal output is separate. Old numerical results are not expected to reproduce under the revised model.

The five historical geometry normalizers cancel in reciprocal column normalization. They remain available as diagnostic reference scales; they are not five independent uncertainty models and are not claimed to be physical maximum bond angles. Consensus relaxation replaces the old Q-learning label and reports the actual residual. There is no reinforcement-learning agent in this implementation.

## Installation and quick start

Python >=3.9 and NumPy >=1.24 are declared. Installed wheels are tested locally on Linux with Python 3.12/NumPy 2.3.5 and Python 3.9/NumPy 1.24.0. Windows and macOS are additional CI targets; consult the public workflow results for their current outcomes. NumPy is the only mandatory dependency. From this directory:

```bash
python -m pip install .
python -m molfuzzy run examples/lean_energy_scenario.json
python -m molfuzzy run examples/lean_energy_scenario.json --sequence --json --output result.json
python -m molfuzzy sensitivity examples/lean_energy_scenario.json --json
python -m molfuzzy robustness examples/lean_energy_scenario.json --trials 100 --seed 42 --json
```

The installed `molfuzzy` command is equivalent to `python -m molfuzzy`. `--json` writes JSON only to stdout; errors use stderr and exit code 2. `--output` exports the complete result and configuration. Only finite JSON values and unique keys are accepted.

```python
import json
from molfuzzy import DecisionPipeline

with open('examples/lean_energy_scenario.json', encoding='utf-8') as f:
    scenario = json.load(f)
result = DecisionPipeline(
    scenario['criterion_labels'], scenario['alternative_labels'],
    criterion_types=[1, 1, 1, 1],  # +1 benefit, -1 cost
    include_sequence=True,
).run(scenario['criteria_matrices'], scenario['decision_matrices'])
print(result.summary())
```

For the supplied illustrative data, the preference order is **SMRT > SCH > AI > MNT > JTM**. The separate exact angular traversal is **JTM -> SCH -> MNT -> SMRT -> AI**. A traversal's first node is not the best alternative. Scores are model outputs, not measured manufacturing performance.

## Input contract

A scenario has `criterion_labels`, `alternative_labels`, `criteria_matrices`, and `decision_matrices`. Optional JSON fields are `criterion_types`, `expert_weights`, and `criterion_weights`. Labels are unique nonempty strings. Both matrix collections contain the same positive number of experts, in matching expert order. Criterion matrices are exactly C x C; decision matrices are exactly A x C. Diagonals are null (also accepted by the API: empty string or `-`); missing off-diagonal and decision cells are rejected. Extra rows/columns are errors. Values are N/L/M/S/H or their case-sensitive full names. No implicit imputation is performed.

| Term | Assessment | Membership | Non-membership | Hesitancy |
|---|---:|---:|---:|---:|
| Negligible | 1 | 0.20 | 0.70 | 0.10 |
| Low | 2 | 0.40 | 0.50 | 0.10 |
| Moderate | 3 | 0.60 | 0.30 | 0.10 |
| Significant | 4 | 0.80 | 0.15 | 0.05 |
| High | 5 | 0.95 | 0.05 | 0.00 |

This inherited scale is a modeling convention, not an empirically calibrated interval scale. Every MFV has finite components in [0,1] and sums to one within 1e-8 by default. Accepted rounding drift is normalized. Weights must be nonnegative and sum to one; zero total is rejected. A single expert, criterion, or alternative is supported.

## Model specification

1. **Reliability.** Assessment numbers are divided by 5, preserving absolute level differences. Default agreement is `r_e = 1 - mean_offdiag(abs(X_e - mean_experts(X)))`. This is permutation-invariant panel agreement, not a measure of truth or professional competence. `w_e = r_e / sum(r)` is the exact solution of the former unconstrained deviation LP; SciPy and the redundant solver are removed. Supply independent expert weights or `ReliabilityDimensions` when justified.
2. **Consensus.** The target is `T = sum_e w_e M_e`. Convex relaxation `M_e(t+1) = (1-lambda) M_e(t) + lambda T` keeps triples on the simplex. Stop when maximum target residual <=1e-6, otherwise report the cap. The analytic target is used downstream, so learning rate affects only diagnostic convergence speed. It does not provide an independent ranking model. Consensus does not establish correctness or erase the need to examine expert disagreement.
3. **Criterion network.** Full rows retain common criterion coordinates; the diagonal is a zero relation vector. Pairwise angles use the normalized-vector sum/difference formula. Off-diagonal similarity is `1/max(theta,epsilon)`, with epsilon=1e-6 radians. Column normalization produces NR. `x(t+1)=sigmoid(NR @ x(t))`, starting at ones, stops on the fixed-point residual <=1e-8. The state is normalized to criterion weights. These are similarity-network weights, not causal effects. Explicit `criterion_weights` can override them; both network estimates and used weights are retained in the output.
4. **Preference.** Expert decision matrices are reliability-weighted. Benefit utility is `(m-n+1)/2`; cost utility is its complement. Alternative scores are `sum_j w_j u_ij`. Higher is preferred; tie groups use an absolute tolerance of 1e-10 measured from the group's highest score. Lexical order inside a tie is presentation only. Nonnegative fixed weights imply monotonicity; the model assumes additive compensation and cardinal interpretation of the scale. Hesitancy is retained in vectors but has no separate decision penalty in this utility model.
5. **Optional traversal.** Weighted alternative vectors produce the same regularized NR edge costs. The objective minimizes total edge cost of an open Hamiltonian path. Small cases up to 8 alternatives use exhaustive enumeration, with `optimality_proven=True`. Larger cases use an Ant-System-inspired search (alpha=1, beta=2, rho=0.5, initial trail=0.1, max 200 iterations, min 20, patience 30). Inverse-cost deposits are normalized by the best ant fitness each iteration. Log-space transition probabilities avoid exponent overflow. A heuristic patience stop does not prove optimality. `exact_threshold=0` forces heuristic operation. Full logs are opt-in; normal memory is quadratic apart from iteration scalar history. Canonical label ordering stabilizes seeded traversal under row reordering; renaming labels may change tie choices.

For any positive geometry denominator d, `(d/theta_ij)/sum_k(d/theta_kj)` is independent of d. Regularization is applied in raw radians and NR is evaluated in its equivalent canonical form to avoid spurious roundoff sensitivity.

## APIs and diagnostics

- `DecisionPipeline(..., expert_weights=..., reliability_dimensions=..., criterion_weights=...)`
- `rank_alternatives(matrix, weights, labels, criterion_types=...)`
- `balance_experts(matrices, weights, learning_rate=..., tolerance=..., max_iterations=...)`
- `aco_sequence(matrix, weights, labels, exact_threshold=..., keep_log=...)`
- `run_sensitivity_analysis(..., seeds=..., include_sequence=...)`
- `rankings_are_stable(rows, target='alternatives'|'criteria'|'sequence')`
- `perturbation_analysis(...)`: leave-one-expert-out and one-cell adjacent-level perturbations; these are local diagnostics, not confidence intervals.

In automatic-weight mode, the pipeline fails if the criterion network does not meet its fixed-point tolerance; supplied criterion weights remain usable with a flagged nonconverged network diagnostic. A capped consensus diagnostic is returned with `converged=False`; the valid analytic target is still used for scoring, and its status is printed. `to_dict()` includes intermediate matrices, utilities, scores, tie groups, configuration, and optional traversal.

## Validation and reproduction

```bash
python -m pip install '.[dev,plot]'
python -m pytest -q
python docs/reproduce.py
python docs/make_figures.py
python -m build
```

`validation/results.json` contains full case outputs, 15 geometry/rate cases, 100 perturbation trials, three leave-one-out cases, 100 heuristic seeds, and measured host timings. `validation/pytest.*` retains the supplied historical test run; current rebuilt-wheel test logs are in `release-validation/`. See `docs/AUDIT_RESPONSE_zh.md` for all 14 audit responses and `CHANGELOG.md` for migration. Tests cover all 24 criterion permutations, expert/alternative permutations, utility dominance and cost direction, near-duplicate vectors, invalid inputs, convergence caps, CLI JSON, and exhaustive traversal comparison. Passing them supports implementation properties, not domain validity.

The optional `examples/interop_pymcdm_example.py` adapter requires `pip install '.[interop]'`. It demonstrates TOPSIS comparison; agreement is not proof of correctness and disagreement is not a bug in either method. The optional adapter is outside the recorded core test run.

## Scope and publication status

The reviewed 2.0.10 source is hosted at https://github.com/fdhdfgfdfdg/MolFuzzy. The version is prepared for public release on 3 November 2026. Permanent archival registration for this version is pending. The public upstream repository contains version 1.0.0 and must not be cited as the 2.0.10 release. Current authors and support contacts are recorded in AUTHORS.md, CITATION.cff, codemeta.json and package metadata. `CITATION.cff` identifies the software and version without claiming a published SoftwareX article. The case matrices are retained from v1 as an illustrative fixture; their original empirical provenance is not independently verified. No external adoption, operational manufacturing benefit, superiority over competing libraries, or broad clinical suitability is claimed.

## 2.0.1 safety and compatibility changes

- CLI input files can never be output targets, including symlink/hardlink aliases and `--force`. Existing other outputs require `--force`. Symbolic-link output destinations are rejected. A complete temporary file is flushed before atomic publication; errors leave the previous destination intact. Atomic no-clobber output requires hard-link support on the destination filesystem and otherwise fails safely.
- Registered geometry names cannot be overwritten. Raw angles are computed independently of the geometry registry.
- With explicit criterion weights, a nonconverged network remains a diagnostic and does not block scoring. Automatic network-weight mode still requires convergence.
- Normalized similarities use a column-scaled reciprocal expression without forming large reciprocals. `angle_floor` must be finite and at least `sqrt(float64.tiny)` (approximately 1.49e-154 radians), protecting downstream inverse-cost diagnostics. The standalone `reciprocal_matrix` requires epsilon >= `float64.tiny` because it returns actual reciprocals.
- Numeric array APIs reject complex, boolean and textual values before float conversion. Labels may be lists, tuples or NumPy arrays; perturbation analysis accepts nested tuples as well as lists.
- Constructor inputs are defensively copied; result labels and configuration are independent snapshots. Direct modification of pipeline attributes is not a supported configuration interface; construct a new pipeline for a new configuration.
- `parameter_sensitivity_analysis` in `molfuzzy.sensitivity` accepts criterion-weight, expert-weight and angle-floor grids. It reports scores, tie groups, adjacent score gaps and baseline agreement for one-factor scenarios. Weight grids override the corresponding baseline weighting method.

### Installed-distribution validation

Build with `python -m build`, install the resulting wheel in a clean environment with NumPy and pytest, and run `python docs/verify_installed.py` with that environment's interpreter. The script copies tests and examples to an external temporary directory, verifies the imported module is installed in site-packages, runs the complete test suite and checks CLI JSON. CI supplies Linux/Windows/macOS and Python 3.9/3.12 jobs, including NumPy 1.24.0 on Linux/Python 3.9; CI configuration alone is not evidence that those remote jobs ran.

The recorded local release runs use Linux/Python 3.12 with NumPy 2.3.5 and an isolated Linux/Python 3.9 environment with NumPy 1.24.0. See `release-validation/*wheel-tests.txt` in the complete ZIP for current installed-wheel outcomes. New sensitivity evidence is in `validation/results.json` under `parameter_sensitivity`; one baseline and 20 one-factor scenarios are recorded. Simplex-vertex scenarios are stress tests, not probability statements.


## 2.0.4 diagnostics and reproducibility

All sensitivity records now include criterion/alternative labels, the effective configuration, and diagnostics separating valid scores from network/consensus convergence. Parameter records include full configuration snapshots; perturbation trials include diagnostic states, scores, changed cells and input hashes. Failed subset runs are reported, not counted as successful trials. Input validation or a failed baseline still raises an error.

`perturbation_analysis(..., include_perturbations=True, include_leave_one_out=True)` supports independent components. Custom reliability dimensions are sliced consistently when an expert is removed, and their weights are recalculated. A subset with no remaining positive reliability is skipped with a reason. Each component reports success, partial failure or skipped status; successful/failed trial counts define the aggregation denominator. No statistical confidence interval is implied.

`keep_log` accepts Python bool only. `normalize_angle` accepts actual angles in radians in [0, pi] and rejects nonfinite/unrepresentable results. Its output can exceed one for a reference divisor below the angle; geometry divisors are reference scales, not bounds.

Pipeline configuration includes resolved ACO defaults even when traversal is disabled, raw supplied reliability dimensions, used expert weights, the input SHA-256 and runtime versions. The input hash covers ordered labels and matrices serialized as sorted-key compact UTF-8 JSON, retaining original term spellings; it is a content identity check, not a signature or semantic-equivalence guarantee. Model options are recorded separately and do not affect this input-only hash. Input data must be retained alongside outputs for full reproduction.

### Install for the intended task

Core execution only:

```bash
python -m pip install .
python -m molfuzzy run examples/lean_energy_scenario.json --json
```

Full tests and figure regeneration:

```bash
python -m pip install ".[dev,plot]"
python -m pytest -q
python docs/reproduce.py
python docs/make_figures.py
```

The additional 2.0.4 audit mapping is in `docs/AUDIT_RESPONSE_2.0.4_zh.md`. Current results and test counts are in `validation/`; older audit mappings describe their original release scope.


## Version 2.0.4: failure-aware scans and strict export

`parameter_sensitivity_analysis` and `run_sensitivity_analysis` retain their
list return and fail-fast behavior by default (`on_error="raise"`). Opt into
`on_error="record"` to receive a report dictionary with `records`, `requested`,
`successful`, `failed`, `unexecuted`, and `status`. Each record has a status;
failed records retain the parameter value, exception type, reason, and failure
kind (`invalid_parameters` for ValueError, `execution_failure` for RuntimeError,
including network nonconvergence). The parameter report includes its baseline
in the counts. Invalid baselines still raise because comparisons require one.
The generic learning-rate/geometry/seed sweep has no separate baseline.
Unexpected programming exceptions are not suppressed.

```python
from molfuzzy.sensitivity import parameter_sensitivity_analysis
report = parameter_sensitivity_analysis(**scenario,
    angle_floors=[1e-6, 0, 0.1], on_error="record")
assert report["successful"] == 3  # baseline plus two valid values
assert report["failed"] == 1
assert report["unexecuted"] == 0
assert report["all_scenarios_stable"] is None
```

`all_scenarios_stable` is unknown (`None`) if any scenario fails;
`successful_scenarios_stable` describes only the successful subset.
`rankings_are_stable` accepts a complete generic sweep report but rejects an
incomplete one. Runtime nonconvergence remains an execution failure with its
explicit exception reason; valid scores using user weights retain separate
network diagnostic flags.

Every perturbation record, including failures, retains zero-based cell indices,
criterion/alternative labels, original/new terms, and the mutated input hash.
Expert labels are generated positional labels (`expert_1`, etc.), not supplied
identities. The report's seed, original input, and configuration support replay.

Strict recursive JSON export handles nested NumPy arrays and long-double
scalars. Extended precision is rounded to Python binary64 for JSON; nonfinite
values and overflow are rejected, never emitted as NaN or infinity. Invalid
non-JSON scan parameters are retained as an explicitly tagged representation.
ACO options are validated and normalized even when traversal is disabled.
`sequence_execution_status` distinguishes `disabled` from `completed`.


## 2.0.4 numerical and reporting contract

Internally computed convex aggregates and utility scores use a roundoff allowance
of min(1e-8, 8*n*float64.eps), where n is the number of summed terms. Only
values within that allowance are clipped to [0,1]; larger errors and nonfinite
results raise. External MFV component validation remains strict.

Record-mode scan reports are recursively converted to native JSON types before
return. Extended-precision parameters follow the documented binary64 rounding
policy; failed non-JSON parameter values retain tagged representations.

`adjacent_score_gaps` now follows descending raw scores rather than lexical
presentation within ties. `score_gap_order` and `score_gap_pairs` identify the
order and each pair. `between_rank_group_gaps` reports the minimum score of the
higher group minus the maximum score of the next group, with both label lists.
One tie group yields no between-group gaps. A raw score order does not override
the declared tolerance-based preference groups. This corrects gap semantics
when ties occur; consumers must use the new order metadata.


## 2.0.6 consensus and release checks

Consensus targets and every relaxation update are converted to bounded,
normalized MFVs before computing residuals. Returned matrices and residuals
therefore describe the same numerical state; external validation stays strict.

Build releases with `python docs/build_release.py` after installing dev extras.
It builds in a new temporary checkout, verifies metadata and source identity,
and replaces the distribution directory with exactly the current wheel and
sdist. Use `python docs/verify_release.py --archive /path/to/release.zip` after
packaging to verify complete checksum coverage, current distribution names,
metadata and module bytes. Historical audit notes remain documentation only.

### Version 2.0.6

Reliability-dimension weighted sums now receive bounded internal roundoff correction, while external bounds remain strict. Consensus updates use array normalization and construct MFV objects only for the final result. Release verification compares complete recursive Python-module inventories in the source tree, wheel and sdist before comparing bytes.

## Version 2.0.7 reliable normalization at tiny magnitudes

Reliability dimensions remain finite binary64 values in [0,1]. Weighted products use binary exponent scaling. When an active product approaches subnormal magnitude, a standard-library Fraction fallback computes exact products and sums of the supplied binary64 values, then rounds the normalized weights to binary64. Zero beta dimensions do not contribute. A truly zero weighted total is still rejected. Ordinary direct reliability normalization scales by the largest score first.

For independently supplied dimensions, `R` retains the raw weighted reliability rounded to binary64; `R_scaled` is relative to the maximum. `reliability_scale` stores that maximum as mantissa times 2 raised to exponent2, so the scale itself need not fit in binary64. `raw_reliability_underflow` flags positive raw totals displayed as zero. `scaling_method` identifies the calculation path. Weights are calculated before raw diagnostic rounding; callers must not recompute them from rounded R. Final weights smaller than binary64 can represent may still round to zero. Input values already rounded to zero before entering the API cannot be recovered.

## Version 2.0.8 release documentation and validation

The current README title, package version and distribution descriptions are checked together. Historical changelog headings retain their original version numbers. The verifier requires wheel and sdist descriptions to equal the current README, and the sdist README to match the source.

| Platform | Python | NumPy | Evidence status |
| --- | --- | --- | --- |
| Linux | 3.12 | 2.3.5 | Installed-wheel suite and CLI passed; release-validation/current-wheel-tests.txt |
| Linux | 3.9 | 1.24.0 | Installed-wheel suite and CLI passed; release-validation/lower-wheel-tests.txt |
| Windows | 3.9 and 3.12 | Resolver-selected | See public CI results; bundled wheel logs cover Linux |
| macOS | 3.9 and 3.12 | Resolver-selected | See public CI results; bundled wheel logs cover Linux |

The workflow supports manual dispatch and retains logs plus actual runtime versions as job artifacts, including on failure. See docs/CROSS_PLATFORM_VALIDATION.md for execution and acceptance criteria. Configuration alone is not a successful test result.

## Version 2.0.9 citation and evidence packaging

CITATION.cff identifies version 2.0.9 and preparation date 2026-10-22; public publication remains pending. Release verification checks its version and ISO date, and compares its archived bytes.

The sdist contains the frozen validation/ evidence snapshot, including EVIDENCE_SCOPE.md. Numerical results retain their recorded 2.0.7 provenance; earlier installed-wheel logs are explicitly historical. Current source tests are recorded in validation/source-209.txt. The complete ZIP additionally carries release-validation/ with current installed-wheel and packaging results produced after the distributions are built. These post-build logs are deliberately outside the sdist to avoid a self-referential build. The wheel supplies executable code and package metadata, not the evidence bundle.

CI builds to distributions/ and runs docs/verify_release.py before installation tests. The gate checks source and sdist citation metadata and the complete validation/ file inventory and bytes. CI logs remain in Actions artifacts, not in the frozen source snapshot.

## Version 2.0.10 strict citation parsing

Release checks parse the complete CITATION.cff using PyYAML SafeLoader, reject duplicate mapping keys and malformed YAML, and require an exact version and a complete canonical YYYY-MM-DD date. Dates remain strings during parsing so suffixes, timestamps and noncanonical spellings cannot pass through partial matching. This checks YAML syntax and release fields, not the full Citation File Format schema.

PyYAML>=6 is a development/release-check dependency (already in the dev extra), not a core runtime dependency. Install with `python -m pip install ".[dev]"` before running the full tests or `python docs/verify_release.py`. CI installs PyYAML explicitly.

Current 2.0.10 source tests are in validation/source-210.txt; current installed-wheel logs are in release-validation/ in the full ZIP. Older evidence retains its original version and scope.


## Current authors and support

Ruobing Wang (corresponding author): ruobing2@alumni.cmu.edu,
https://orcid.org/0009-0006-7883-6248.
Xingzhi Wang: xingzhi2@alumni.cmu.edu,
https://orcid.org/0000-0001-8895-0809.
Full affiliations, contributions and upstream credit are in AUTHORS.md.
For a support request include the software version, Python/NumPy versions,
operating system, command, minimal input and expected/actual output.

## Current reproducibility record

The numerical record and figures were regenerated for 2.0.10 on 3 November 2026
with Python 3.12.14 and NumPy 2.3.5. The base ranking, 80/100 unchanged full
rankings and 77/100 optimal heuristic costs reproduce the supplied results.
Current timings belong to this host and are not a speed comparison with older
measurements. EVIDENCE_SCOPE.md distinguishes current and historical records.
The current runtime lock files in requirements-validation-py312.txt and
requirements-validation-py39.txt record exact validation dependencies, separate
from the supported lower-bound dependency declaration in pyproject.toml.
The software version is unchanged because the computational modules are
unchanged; this preparation fills author metadata and renews validation evidence.
