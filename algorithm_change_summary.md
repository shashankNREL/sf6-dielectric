# Algorithm Change Summary

This document describes the algorithm and code changes made in `/home/runner/work/sf6-dielectric/sf6-dielectric/sf6_pareto_design.py`, along with the assumptions used when making those changes.

## Files changed

- `/home/runner/work/sf6-dielectric/sf6-dielectric/sf6_pareto_design.py`
- `/home/runner/work/sf6-dielectric/sf6-dielectric/.gitignore`

## Why the changes were made

The main goal was to make the prototype more internally consistent and less likely to reward surrogate/modeling errors during NSGA-II search. The changes focused on:

- fixing target semantics,
- making invalid molecules truly infeasible,
- improving surrogate validation,
- exposing prediction uncertainty,
- strengthening applicability-domain checks,
- improving Pareto candidate ranking and active-learning selection,
- cleaning up a small repository hygiene issue.

## Detailed code changes

### 1. Target semantics were made consistent

The most important fix was to make `gwp100` use one consistent meaning throughout the pipeline.

#### What changed

- Added per-target transform metadata in `SurrogateEnsemble.TARGET_TRANSFORMS`.
- Moved GWP transformation logic into:
  - `_transform_target(...)`
  - `_inverse_transform_target(...)`
- Kept `ds_rel` and `bp_c` on their raw scales.
- Trained the GWP surrogate on `log10(gwp100)` internally, but returned predictions to the rest of the pipeline on the raw GWP scale.
- Updated the optimizer so:
  - the third objective is `log10(raw_gwp)`,
  - the GWP constraint is enforced on raw GWP (`gwp <= 5000`),
  - decoded/exported predictions use raw GWP.

#### Why

Previously the model was trained on log-GWP but optimization logic treated the prediction inconsistently. That allowed the objective and constraint to use different semantics. The new version makes the transform explicit and centralized.

---

### 2. Invalid molecules were made constraint-violating

#### What changed

- Added `INVALID_CONSTRAINT_PENALTY = 1e3`.
- In `SF6ReplacementProblem._evaluate(...)`, invalid molecules and feature-generation failures now receive:
  - poor objective values, and
  - large positive constraint violations.

#### Why

`pymoo` treats inequality constraints as satisfied when `g <= 0`. The earlier behavior allowed invalid molecules to look feasible from the constraint standpoint. The new behavior makes them explicitly infeasible.

---

### 3. Single models were replaced with bootstrap ensembles

#### What changed

- `SurrogateEnsemble` now trains multiple gradient-boosting pipelines per target instead of one model per target.
- Added:
  - `n_members`
  - `_base_estimator(...)`
  - `predict_with_uncertainty(...)`
  - `predict_single_with_uncertainty(...)`
- Each member is trained on a bootstrap resample of the training set.
- Prediction means are used as the surrogate outputs.
- Prediction standard deviations are exported as model-disagreement uncertainty.

#### Why

This keeps the original gradient-boosting approach but adds a practical uncertainty signal without changing the overall architecture.

---

### 4. Validation was strengthened and leakage was reduced

#### What changed

- Removed the previous pattern where scaling was fit before cross-validation.
- Cross-validation now fits preprocessing inside each fold through a pipeline.
- Replaced CV reporting that only returned average R² with fold-safe out-of-fold metrics:
  - `R²`
  - `MAE`
  - `RMSE`
  - Spearman rank correlation
- Added `holdout_evaluate(...)` for a simple train/test split.
- `main()` now prints both CV and holdout metrics.

#### Why

This does not solve the small-data problem, but it does reduce optimism caused by preprocessing leakage and gives a fuller view of surrogate quality.

---

### 5. Applicability-domain logic was strengthened

#### What changed

- Added training-set scaling storage:
  - `feature_scaler`
  - `X_train_scaled`
- Added nearest-neighbor support with `NearestNeighbors`.
- Replaced a leverage-only boolean check with `applicability_domain_details(...)`, which returns:
  - `in_domain`
  - leverage
  - leverage threshold
  - nearest-neighbor distance
  - distance threshold
- `applicability_domain(...)` now wraps the richer method.
- Candidate decoding now exports these AD diagnostics.

#### Why

Leverage alone was too weak for this search setting. The new version is intentionally conservative by requiring both leverage and local-distance checks to pass.

---

### 6. Pareto decoding and exports were enriched

#### What changed

- `decode_population(...)` now includes:
  - `ds_std`
  - `bp_std`
  - `gwp_std`
  - AD diagnostics
  - feasibility flag based on the actual constraint vector
- Exported Pareto candidates therefore contain both predicted properties and trust/risk metadata.

#### Why

This makes downstream review less dependent on raw objective values alone.

---

### 7. Candidate ranking became uncertainty-aware

#### What changed

- Added:
  - `normalize_series(...)`
  - `add_candidate_priority_scores(...)`
- Hypervolume contribution is still used, but ranking now also considers:
  - in-domain status,
  - relative model uncertainty.
- Added named weight dictionaries:
  - `PRIORITY_SCORE_WEIGHTS`
  - `ACTIVE_LEARNING_SCORE_WEIGHTS`
- `main()` now ranks the feasible set with the new priority score instead of hypervolume alone.

#### Why

The intent was to reduce “surrogate optimum chasing” by preferring candidates that are strong on the frontier while also being more trustworthy.

---

### 8. Active-learning scoring was updated

#### What changed

- `active_learning_round(...)` was updated to score candidates using a blend of:
  - Pareto value,
  - novelty,
  - uncertainty,
  - in-domain status.
- Novelty scoring against the training set was vectorized instead of computed one candidate at a time.
- Feature-extraction failures are no longer treated as if they sit at the origin of feature space.

#### Why

The previous active-learning logic was still a sketch. The new version is still lightweight, but it is more aligned with uncertainty-aware selection.

---

### 9. Console output and saved results were updated

#### What changed

- Training logs now show stronger validation metrics.
- The top-candidate table now includes uncertainty and applicability-domain status.
- The printed top candidate now includes DS/BP/GWP uncertainty.
- `output/pareto_candidates.csv` now contains richer ranking and trust metadata.

#### Why

The pipeline output now better reflects confidence and support, not just predicted performance.

---

### 10. Small code-quality cleanups were made

#### What changed

- Removed several unused imports and dead code paths that were no longer needed.
- Removed the global warnings suppression line.
- Removed a tracked bytecode artifact and added:
  - `/home/runner/work/sf6-dielectric/sf6-dielectric/.gitignore`
  - ignore entries for `__pycache__/` and `*.pyc`

#### Why

These were small hygiene changes that reduce noise and avoid committing generated files.

## Assumptions made when implementing the changes

### A. GWP should remain user-facing on the raw scale

Assumption:

- The best compromise was to train GWP on a log scale for stability, but keep optimization constraints, reporting, and exports on the raw GWP scale except where a log-scale objective is explicitly intended.

Reason:

- This preserves the intended search behavior while making the code easier to reason about.

### B. Ensemble disagreement is an acceptable first uncertainty proxy

Assumption:

- Bootstrap ensemble spread is a reasonable approximation of epistemic uncertainty for this prototype.

Reason:

- It is lightweight, uses the existing sklearn stack, and does not require a larger modeling rewrite.

### C. Conservative applicability-domain filtering is preferable

Assumption:

- It is better to reject some potentially good candidates than to over-trust extrapolative surrogate predictions.

Reason:

- The dataset is very small relative to the search space, so false confidence is a bigger risk than over-filtering.

### D. The existing GBR-based architecture should stay in place

Assumption:

- The changes should improve reliability without replacing the whole surrogate/search framework.

Reason:

- The stated objective was to harden the current prototype rather than redesign it into a new system.

### E. The dataset itself was not rebuilt in this change set

Assumption:

- Data expansion, provenance cleanup, family-aware splits, and physics-rich labels are necessary future steps, but were intentionally left out of this implementation pass.

Reason:

- Those changes require new source data and a broader redesign, not just code edits.

### F. The active-learning function remains heuristic

Assumption:

- The revised active-learning scoring is still a heuristic ranking rule, not a calibrated acquisition function.

Reason:

- It improves over the earlier sketch, but it is not yet a full Bayesian or conformal active-learning workflow.

### G. Weight values are heuristic defaults

Assumption:

- The priority and acquisition weights are intended as reasonable defaults, not scientifically optimized constants.

Reason:

- They were chosen to balance Pareto quality, trust, and uncertainty without introducing a more complex tuning framework.

### H. Full runtime validation was not available in the sandbox

Assumption:

- Syntax validation was possible, but end-to-end execution was blocked locally by missing Python dependencies in the environment.

Reason:

- The pipeline baseline failed locally because `numpy` was not installed in the sandbox environment.

## What these changes do not solve

These changes improve reliability, but they do **not** solve the biggest structural limitations of the prototype:

- the dataset is still very small,
- dielectric-strength features are still mostly proxy-based,
- the optimization problem is still only three properties,
- the search still focuses on pure compounds,
- the generator still favors validity more than chemical realism,
- the system is still a single-script prototype rather than a modular pipeline.

## Recommended next documentation/use

If this repository continues to evolve, the next useful documents would be:

- a dataset provenance and curation note,
- a model-validation note with family-aware splits,
- a design note for mixture optimization,
- a roadmap for modularizing the single-file prototype.
