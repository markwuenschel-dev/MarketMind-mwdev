Adaptive Test Infrastructure
============================

Core framework for self-evolving, scenario-driven testing. This README explains how the pieces fit, how scenarios flow from generation to execution, and how to extend the system safely.

* * * * *

Table of Contents
-----------------

1.  [Execution Pipeline](#execution-pipeline)

2.  [Override System](#override-system)

3.  [Learning Engine](#learning-engine)

4.  [Supporting Utilities](#supporting-utilities)

5.  [End-to-End Flow](#end-to-end-flow)

6.  [Scenario Data Contract](#scenario-data-contract)

7.  [Capabilities & Probes (`requires`)](#capabilities--probes-requires)

8.  [Fingerprinting, Dedupe & Stability](#fingerprinting-dedupe--stability)

9.  [Snapshots & Drift](#snapshots--drift)

10. [Skip / XFail Policy](#skip--xfail-policy)

11. [Override DSL Quick Reference](#override-dsl-quick-reference)

12. [Extension Points](#extension-points)

13. [CLI & Flags](#cli--flags)

14. [Phase Progression & Milestones](#phase-progression--milestones)

15. [Quick Start](#quick-start)

16. [Troubleshooting](#troubleshooting)

17. [Glossary](#glossary)

* * * * *

Execution Pipeline
------------------

-   `adaptive_executor.py` --- **Registry, classification, handler dispatch.** Maps a typed scenario to a concrete handler; provides classification utilities for telemetry and routing.

-   `adaptive_matrix.py` --- **Scenario parametrization & fingerprinting.** Normalizes raw dicts to Pydantic models, computes fingerprints, applies capability probes, and yields pytest params in a deterministic order.

-   `scenario_models.py` --- **Pydantic models for type safety.** Strict kinds (fail on extra fields) and an `EnsembleScenario` (forward-compatible, allows extras). Accepts `requires` for pre-param pruning.

Override System
---------------

-   `override_engine.py` --- **DSL interpretation & module patching.** Converts passthrough overrides (e.g., `module:`, `param:`, `special:`) into a concrete, ordered **override plan** with dotted-attr blocking and overwrite tracking.

-   `scenario_policy.py` --- **Skip policy, normalization, ENV passthrough.** Additive helpers that run before execution.

Learning Engine
---------------

-   `self_evolving_engine.py` --- **Adaptive learning, parallel decisions.** Emits engine-aware scenarios (e.g., threshold edges, optimal sequences, high-risk patterns) and can tag volatility for skip-learning.

Supporting Utilities
--------------------

-   `compat_layer.py` --- **Environment capability detection.** Surface `gpu_present`, `parallel_exec_supported`, OS family, file system flags, etc.

-   `compat_adapter.py` --- **Capability attachment helpers.** Attaches computed capabilities to the engine/executor, so probes can read them.

* * * * *

End-to-End Flow
---------------

`┌────────────┐      ┌────────────────┐      ┌─────────────────┐      ┌──────────────┐
 │ generator  │ ---> │ normalization  │ ---> │ override plan   │ ---> │ matrix       │
 │ (static +  │      │ (schema gate:  │      │ (_override_plan)│      │ (probes,     │
 │ learned)   │      │  overrides_norm│      │ + errors)       │      │  fingerprint)│
 └────────────┘      └────────────────┘      └─────────────────┘      └─────┬────────┘
                                                                            │
                                                                  ┌─────────▼─────────┐
                                                                  │ executor          │
                                                                  │ (classify, route, │
                                                                  │  handler run)     │
                                                                  └───────────────────┘`

**Key invariants**

-   **Single contract gate:** All scenarios first pass a schema normalizer. Unknown/extra keys do not vanish; they become **passthrough** for the DSL to interpret into an **override plan**.

-   **Plan persistence:** `_override_plan` is persisted on each scenario for auditability and deterministic application.

-   **Pre-param pruning:** `requires` lets the matrix drop cases early if environment capabilities don't match.

* * * * *

Scenario Data Contract
----------------------

Each scenario (raw dict or model) adheres to these fields. Strict kinds enforce shape; `EnsembleScenario` allows extra keys for forward-compat.

**Common fields**

-   `kind: str` --- Scenario type (e.g., `parallel_threshold_plus`, `optimal_sequence`, `ensemble_*`).

-   `overrides: Dict[str, Any]` --- Author-intent overrides (e.g., `{"config.mode":"fast","debug":False}`).

-   `overrides_normalized: Dict[str, Any]` --- Post-schema mapping to harness injection keys (e.g., `{"config.mode":"fast","runtime.debug":false}`).

-   `_override_plan: Dict[str, Any]` --- Ordered plan produced by the DSL from passthrough keys; carries `apply`, `blocked`, `overwrites` lists.

-   `requires: Dict[str, Any]` --- Capability gates (e.g., `{"gpu_present": true, "parallel_exec_supported": true}`).

-   `expectations: Dict[str, Any]` --- Assertions, marks, advisory metadata; may include `snapshot`.

-   `name: str` --- Stable human name (`config.mode=fast & debug=False`).

-   Aux: `_schema_errors`, `_is_combo`, `_negative_space`, `_risk_score`, `fingerprint`, `_predicted_type` (telemetry), etc.

* * * * *

Capabilities & Probes (`requires`)
----------------------------------

**Purpose:** Avoid parametrizing cases that cannot run in the current environment.

-   Authors (or the generator) set `requires`, e.g.:

    `{ "requires": { "gpu_present": true, "parallel_exec_supported": true } }`

-   The **matrix probe wrapper** reads `requires`, consults `engine_obj.caps`, and **drops** mismatching cases before pytest parameter creation.

-   Keep probes **fast, deterministic, and side-effect free.**

**Common capabilities**

-   `parallel_exec_supported` --- Thread/process/vectorization available.

-   `gpu_present` --- CUDA/ROCm device available.

-   `os_family` --- e.g., `linux`, `darwin`, `windows`.

-   `fs_features` --- e.g., `case_sensitive: true`.

* * * * *

Fingerprinting, Dedupe & Stability
----------------------------------

-   **Stable fingerprint** over the typed model snapshot enables:

    -   **Dedupe:** Avoid duplicate params from different sources (static vs learned).

    -   **Triage:** CI can key results to a stable ID.

-   **Deterministic sort** (by `kind`, `name`, fingerprint, origin) makes param order stable across machines.

* * * * *

Snapshots & Drift
-----------------

-   Optional snapshot capture stores a sanitized view of harness outputs (e.g., `returncode`, cleaned `stdout`/`stderr`).

-   **Drift** compares the new snapshot versus the committed one and annotates the scenario (`_behavior_drift`) for review.

-   With `_override_plan` persisted, drift is **explainable** (plan change vs environment vs logic).

* * * * *

Skip / XFail Policy
-------------------

-   **Learned skip cache:** Noisy cases can auto-skip after N failures (`min_fail_skip`).

-   **Advisory marks:** High-risk/unstable cases can be tagged to avoid blocking merges while preserving coverage signal.

-   **Unknown handler behavior:** Classifier determines `unknown`, `known/no handler`, `handler present`; policy maps that to xfail/skip/execute.

* * * * *

Override DSL Quick Reference
----------------------------

The DSL turns passthrough keys into a **concrete plan**:

-   `module:.*` --- Target a module or class for patching.

    -   Example: `module:strategy.ma = "EMA"`

-   `param:.*` --- Constructor or runtime parameter injection.

    -   Example: `param:zero_on_any_nan = True`

-   `special:*` --- Non-standard actions (block dotted attrs, apply hooks, etc.).

    -   Example: `special:block = "self.risky_attr"`

**Plan shape (example)**

`{
  "apply": [
    {"module":"strategy.ma","param":"zero_on_any_nan","value":false},
    {"special":"block","target":"self.risky_attr"},
    {"param":"ma_type","value":"ema"}
  ],
  "blocked": ["self.risky_attr"],
  "overwrites": [{"key":"param.ma_type","old":"sma","new":"ema"}]
}`

* * * * *

Extension Points
----------------

-   **New scenario kinds:** Add a Pydantic model in `scenario_models.py`, register in kind → model map, implement a handler in `adaptive_executor.py`.

-   **New capabilities:** Implement detection in `compat_layer.py`, surface on `engine_obj.caps`, and allow `requires` to reference it.

-   **New probes:** Extend the probe wrapper in `adaptive_matrix.py` with fast checks mapped from `requires`.

-   **Generator expansions:** Emit new kinds (`chunk_size_minus/plus`, `op_failure_guard`, anti-sequence), pairwise-budgeted cartesian, and coverage JSON.

* * * * *

CLI & Flags
-----------

> Flags vary by tool; below are common patterns to standardize:

-   **Matrix knobs (pytest via decorators/env):**

    -   `min_fail_skip=<N>` --- After N failures for a stable fingerprint, auto-skip future runs.

    -   `environ_filter=...` --- Reduce matrix explosion by environment tag.

-   **Generator knobs (if using the codegen script):**

    -   `--combos` --- Emit multi-condition branch combos.

    -   `--cartesian --max-cartesian K` --- Risk-ranked cartesian pairs, capped by K.

    -   `--strict-schema` --- Drop scenarios that cannot be normalized.

    -   `--snapshot --snapshot-diff` --- Capture & compare harness snapshots.

    -   `--emit-coverage-json PATH` --- Write branch/interaction coverage telemetry.

* * * * *

Phase Progression & Milestones
------------------------------

-   **Phase 0** ✅ --- Wired executor/matrix baseline\
    *Handlers registered, basic classification, deterministic parametrization, fingerprinting.*

-   **Phase 1A** ⬅️ *YOU ARE HERE* --- Skip policy & normalization, ENV passthrough\
    **Deliverables**

    -   Schema normalization at the gate (`overrides_normalized` persisted).

    -   `_override_plan` persisted for static scenarios.

    -   Learn-skip cache enabled with configurable `min_fail_skip`.

    -   Parallel capability probe implemented; probe wrapper scaffolding in place.\
        **Acceptance**

    -   Failing flaky case transitions to auto-skip after N; persisted plan appears in CI artifacts; unknown keys no longer vanish.

-   **Phase 2** 🔜 --- Engine handler implementation\
    **Deliverables**

    -   Rich engine-aware kinds (`chunk_size_*`, `op_failure_guard`, anti-sequence).

    -   Scenario-level `requires` honored generically in probe wrapper.

    -   `_predicted_type` stamped during normalization for observability.

-   **Phase 3** 🔜 --- Override application fixes & coverage budgeting\
    **Deliverables**

    -   Budgeted pairwise cartesian/negative-space selection.

    -   Snapshot payloads include scenario/metric fingerprints.

    -   Coverage JSON emitted and charted in CI.

* * * * *

Quick Start
-----------

1.  **Author or generate scenarios**

    -   Add raw dicts or generated lists. Keep `overrides` concise; pass extras via DSL passthrough.

2.  **Persist plan & normalization (static)**

    -   Run your `_persist_plan_on_scenario(sc)` over static lists to stamp `overrides_normalized`, `_override_plan`, `_schema_errors`.

3.  **Gate with `requires` (optional but recommended)**

    -   Add `requires` to scenarios when capability assumptions exist (e.g., GPU).

4.  **Run tests**

    -   Pytest picks up parametrized cases from `adaptive_matrix.py` output.

5.  **Review CI artifacts**

    -   On failure/drift, inspect `_override_plan`, `overrides_normalized`, and snapshot deltas.

* * * * *

Troubleshooting
---------------

-   **Scenario rejected by model**

    -   Strict kinds forbid extras; either move to `EnsembleScenario` or update the model to accept the new field.

-   **Case skipped early**

    -   Check `requires` vs `engine_obj.caps`. Probe wrapper may be pruning intentionally.

-   **Drift with no obvious code change**

    -   Compare `_override_plan`; look for new `blocked` entries or `overwrites`. Confirm environment caps didn't change.

-   **Param explosion**

    -   Use generator budgets (`--max-cartesian`) and environment filters. Prefer pairwise coverage to full cartesian.

-   **Unknown handler**

    -   Inspect `_predicted_type` and `kind`; ensure executor registry has a handler mapping.

* * * * *

Glossary
--------

-   **Contract gate** --- The schema normalization that maps author overrides into harness injection keys and types.

-   **Passthrough** --- Unknown keys preserved for the DSL; never silently dropped.

-   **Override plan** --- Deterministic sequence of actions (`apply`, `blocked`, `overwrites`) executed by the override engine.

-   **Probe** --- Fast predicate evaluated before parametrization to drop incompatible cases.

-   **Fingerprint** --- Stable hash over the typed scenario for dedupe and triage.

-   **Drift** --- Snapshot mismatch between current and stored behavior for the same scenario signature.