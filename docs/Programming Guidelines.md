MarketMind
Programming Guidelines v3.1
Version 3.1 · April 2026
Engineering principles for abstract, type-guided, declarative, extensible, combinatoric, pipeline-oriented, parallel, and adaptive system design
Audience: Internal engineering, technical stakeholders
Table of Contents
1. Purpose
2. North-Star Constraints
3. Core Engineering Model
4. Architecture Rules
5. Performance Engineering Rules
6. Testing and Gates
7. Operational Visibility and Failure Discipline
8. Evolution and Adaptation
1. Purpose
These guidelines define how MarketMind code should be written and structured when the goal is: maximum abstraction, type-guided composition, declarative construction, combinatoric exploration, pipeline-oriented execution, parallel scalability, and computational efficiency. The codebase is designed to evolve under non-stationarity; therefore factories, registries, schemas, and explicit execution plans are non-negotiable.
2. North-Star Constraints
The system must remain: point-in-time correct, gate-governed, reproducible within defined determinism tiers, auditable via content-addressed artifacts, and observable in production.
DECISION: Correctness beats speed when they conflict
If a performance optimization undermines PIT integrity, determinism, schema validity, or gate validity, it is rejected. Speed work is allowed only inside the functional core, behind invariant tests and explicit instrumentation.
3. Core Engineering Model
3.1 Abstract and Type-Guided by Default
Design around contracts (Protocol, ABC, typed boundary model), not concrete classes. Implementation types must be swappable without touching callers. Prefer narrow behavioral interfaces, typed value objects, and fixed-shape IRs with masking over ad-hoc polymorphism and deep inheritance.
Runtime flexibility is allowed only behind stable typed contracts.
3.2 Extensible, Registry-Driven Composition
All extensible capabilities register into explicit registries (runtime registries and/or entry points). No static wiring for signals, ops, strategies, allocators, planners, gates, or cost/execution models unless it is a deliberate, measured hot path.
The strategic goal is extensibility. Registries are the mechanism. Extension points must be explicit, testable, and version-aware.
Multiple authoring surfaces may exist temporarily for compatibility, but each subsystem must lower into a single canonical IR/planner/executor path. Parallel execution models for the same semantic layer are transitional only.
3.3 Declarative, Schema-First Construction
Variation should be expressed in manifests, schemas, typed configs, and IR dimensions before it is expressed in imperative branching. Prefer specification over handwritten orchestration.
Boundary objects must be validated at construction time. Invalid states should fail early, structurally, and with typed errors.
DECISION: New variation enters through schema first
If a new capability can be expressed as a manifest dimension, registry entry, or planner input, do that before adding bespoke control flow.
3.4 Factory-First Systems (Prioritized)
Prefer factory code over hand-wired implementations. Factories are the multiplication layer: they generate graphs, plans, pipelines, and combinatoric variants safely. A factory is successful when it can produce large families of valid objects from small, declarative specs (schemas, manifests, IR).
Factory code should focus on: (1) canonical IR construction, (2) validation at boundaries, (3) lowering/compilation to efficient execution plans, (4) stable hashes/keys for reuse, (5) parallel-friendly partitioning (symbols, time blocks, folds, tasks), (6) emission of reproducibility metadata.
DECISION: Factories own combinatorics
Combinatoric search (signals × params × costs × execution policies × regimes) must be expressed as factory outputs, not manual glue code.
3.5 Combinatoric by Design
Every module should be composable: small primitives that can be recombined by a factory. Avoid bespoke one-off pipelines. If a new variant is needed, add it as a manifest/schema dimension, planner rule, or registry entry.
Manual enumeration does not scale and is not a substitute for design.
3.6 Pipeline-Oriented Architecture
Model the system as explicit stages with typed handoff points: specification → IR → validated IR → plan → execution → artifacts → gates.
Stage boundaries must be visible in code and inspectable in artifacts. Do not collapse planning, execution, and evaluation into opaque control flow. Hidden pipelines become untestable and unauditable.
3.7 Batch, Vectorized, and Parallel Execution
Assume large universes, many tasks, and many folds. Prefer columnar/vectorized compute over Python loops; prefer batch ops over per-symbol calls; design functions to be embarrassingly parallel over symbols, folds, and regime episodes.
Write Python as an orchestration language for efficient kernels, not as the default location of per-element compute.
3.8 Efficiency Over Readability (With Guardrails)
Performance and scalability are primary. Dense or non-obvious implementations are acceptable when they are: (1) heavily tested with invariants, (2) instrumented, (3) bounded by contracts and schemas, (4) isolated behind stable interfaces, (5) justified by profiling or benchmark evidence.
Unreadable code is acceptable only in explicit hot paths. Accidental opacity is not an optimization strategy.
4. Architecture Rules
4.1 Functional Core / Imperative Shell
All computational logic (data transforms, features, signal generation, allocators, planners) must be pure and deterministic. Side effects (I/O, networking, broker calls, clocks, persistence) stay in the shell.
The shell coordinates. The core computes.
4.2 Fixed-Dimensional Interfaces with Masking
Prefer fixed-shape tensors/vectors/matrices with masks over dynamic resizing. This supports replay compatibility, caching, planning simplicity, and fast kernels (CPU/GPU).
Use dynamic structure only where it is structurally necessary and outside hot execution surfaces.
4.3 Point-in-Time Discipline
All market, fundamental, and macro data access must flow exclusively through the PIT front door (e.g., DataView.as_of(T)) that rigorously enforces valid_time <= T and knowledge_time <= T. This boundary is essential for correctly handling backfilled, revised, or point-in-time data without introducing lookahead bias.
No direct access to raw datasets is permitted in strategy code, planner logic, feature engineering, or evaluators.
Point-in-Time discipline is not a convention or best practice. It is a non-negotiable, architecturally enforced boundary critical to the integrity of all backtests, live signals, and production deployments.
4.4 Determinism Tiers
Every pipeline declares its determinism tier requirement (D3/D2/D1). Use hierarchical seed derivation, deterministic ordering, explicit partitioning, and stable aggregation. If results are not deterministic within the required tier, they are not promotable.
Determinism requirements belong in pipeline metadata, not tribal memory.
4.5 Value Semantics and Mutation Discipline
Prefer immutable or effectively immutable value objects for manifests, IR nodes, plans, descriptors, and artifact metadata. Mutation must be explicit, local, and isolated from the functional core.
Shared mutable state is a last resort. Hidden mutation across planning or execution boundaries is forbidden.
5. Performance Engineering Rules
5.1 Hot Paths Are Explicit
Performance-critical sections must be isolated and labeled. Use profiling and benchmarks to justify micro-optimizations. Keep hot paths small and stable; keep experimentation in factories, manifests, and planners.
If a section is performance-critical, that fact should be visible in the code and in benchmark coverage.
5.2 Memory and Copy Discipline
Minimize materializations, copies, and conversions. Prefer lazy/streaming execution where feasible. Cache at the right layer: spec/IR → plan → materialized features → execution artifacts.
Cache keys must include registry versions, schema versions, and PIT boundaries. Cached outputs that cannot be named precisely cannot be trusted.
5.3 Parallel Execution Model
Write compute in forms that can map to: multiprocessing, thread pools, vectorized kernels, GPU batches, or distributed schedulers. Partition along stable axes (symbol/time/fold/task) and keep cross-partition communication minimal.
Cross-partition coordination should be explicit, coarse-grained, and rare.
6. Testing and Gates
6.1 Invariants > Examples
Use property-based tests for invariants: leakage, PIT boundaries, determinism, monotonicity, stability bounds, schema/IR round-trips, hash stability, and idempotence of planners/executors.
Examples are useful. Invariants are authoritative.
6.2 Gate-Oriented Development
Every new capability must expose measurable outputs for gates (stat validity, cost realism, meta-validity, PIT, determinism, execution integrity). If it cannot be gated, it is not done.
Factories, planners, and evaluators must emit the evidence required for promotion.
7. Operational Visibility and Failure Discipline
No print statements. Use structured logging, tracing, and explicit execution metadata. Factory outputs must be traceable: every plan has an ID/hash, every execution emits enough metadata to reproduce and debug, and every artifact can be tied back to its inputs, registry state, and determinism tier.
Observability is part of correctness, not post hoc tooling.
7.1 Exception Handling
Error handling must be precise, typed, and domain-aware. Bare except: is forbidden. except Exception: is allowed only for structured logging followed by immediate re-raise, translation to a typed domain error, or controlled boundary handling with explicit policy.
Silent failure paths are forbidden.
CI lint rule: forbid bare except and silent except Exception blocks.
7.2 Reproducibility Metadata
Executions must emit the metadata required to explain a result: manifest/spec hash, registry versions, planner version, seed lineage, PIT boundary, determinism tier, partition identity, and artifact lineage.
If a result cannot be reconstructed from emitted metadata, it is not a production-grade result.
8. Evolution and Adaptation
Expect signal churn, model drift, and regime drift. Design for replacement, not permanence. Factories, schemas, registries, and planners are the evolution surface; gates, typed boundaries, and artifact lineage are the safety surface.
Adaptation must change components without dissolving contracts. Extensibility is mandatory. Structural discipline is non-negotiable.