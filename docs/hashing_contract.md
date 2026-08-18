# Hashing Contract

**Status:** LOCKED  
**Last Updated:** 2025-01-15  
**Owners:** Platform Team  
**Canonical Implementation:** `marketmind/artifacts/hashing_contract.py`

---

## 1. Canonical JSON Profile

| Rule | Specification | Enforcement |
|------|---------------|-------------|
| Serialization | **RFC 8785 (JCS)** | Deterministic key ordering, UTF-8 normalization |
| Parser | `json.loads()` with `parse_float=Decimal` for money fields | Runtime validation |
| Encoder | Custom encoder enforcing all rules below | `HashingContract.canonicalize()` |

---

## 2. Banned Values

| Value | Policy | Error Behavior |
|-------|--------|----------------|
| `NaN` | **ILLEGAL** | Raise `HashingContractViolation` on serialize |
| `Infinity` / `-Infinity` | **ILLEGAL** | Raise `HashingContractViolation` on serialize |
| `None` in required fields | **ILLEGAL** | Raise `HashingContractViolation` on serialize |
| Empty strings in identity fields | **ILLEGAL** | Raise `HashingContractViolation` on serialize |

### Detection

```python
def _check_banned_values(obj: Any, path: str = "") -> None:
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            raise HashingContractViolation(f"Banned float value at {path}: {obj}")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _check_banned_values(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _check_banned_values(v, f"{path}[{i}]")
```

---

## 3. Float Policy (Strict)

### Rule: No Raw Floats in Identity-Critical Artifacts

Floats are **prohibited** in artifacts that contribute to identity hashes unless one of:

1. **Decimal string representation** for money/prices/PnL
2. **Explicit quantization policy** with named ID

### Quantization Policies

| Policy ID | Rule | Use Case |
|-----------|------|----------|
| `Q_NONE` | Floats prohibited; use decimal strings | Prices, PnL, costs |
| `Q_ROUND8` | `round(x, 8)` → format as `f"{x:.8e}"` (scientific notation) | Normalized features |
| `Q_ROUND6` | `round(x, 6)` → format as `f"{x:.6e}"` | Metrics, ratios |
| `Q_INT` | `int(round(x))` | Counts, discrete values |

### Formatting Rules (for Q_ROUND* policies)

```python
def format_float(value: float, policy: str) -> str:
    """Canonical float formatting for hashing."""
    if policy == "Q_NONE":
        raise HashingContractViolation("Floats not allowed under Q_NONE policy")
    
    precision = {"Q_ROUND8": 8, "Q_ROUND6": 6, "Q_INT": 0}[policy]
    
    if policy == "Q_INT":
        return str(int(round(value)))
    
    # Round, then format in scientific notation (deterministic across platforms)
    rounded = round(value, precision)
    
    # Handle negative zero
    if rounded == 0.0:
        rounded = 0.0
    
    # Scientific notation with explicit sign and fixed exponent width
    return f"{rounded:.{precision}e}"
```

### Examples

| Input | Policy | Output |
|-------|--------|--------|
| `123.456789012` | `Q_ROUND8` | `"1.23456789e+02"` |
| `0.00001234567` | `Q_ROUND6` | `"1.234567e-05"` |
| `-0.0` | `Q_ROUND8` | `"0.00000000e+00"` |
| `1500.50` | `Q_NONE` | `"1500.50"` (as decimal string, not float) |

### Artifact Field Classification

| Field Category | Policy | Rationale |
|----------------|--------|-----------|
| Prices, PnL, costs, fees | `Q_NONE` (decimal strings) | Exact representation required |
| Sharpe, Sortino, ratios | `Q_ROUND6` | 6 decimals sufficient for comparison |
| Normalized features, embeddings | `Q_ROUND8` | Balance precision vs. hash stability |
| Counts, bar indices | `Q_INT` | Discrete values |

---

## 4. Exclusion List

Fields **always excluded** from content hashes:

| Field | Reason |
|-------|--------|
| `created_at` | Non-deterministic timestamp |
| `updated_at` | Non-deterministic timestamp |
| `hostname` | Runtime locality |
| `worker_id` | Runtime locality |
| `wall_duration_ms` | Non-deterministic timing |
| `cpu_time_ms` | Non-deterministic timing |
| `gpu_time_ms` | Non-deterministic timing |
| `memory_peak_bytes` | Non-deterministic resource usage |
| `run_id` | Identity reference, not content |
| `artifact_id` | Identity reference, not content |

### Implementation

```python
HASH_EXCLUSIONS: frozenset[str] = frozenset({
    "created_at", "updated_at",
    "hostname", "worker_id",
    "wall_duration_ms", "cpu_time_ms", "gpu_time_ms",
    "memory_peak_bytes",
    "run_id", "artifact_id",
})

def strip_exclusions(obj: dict[str, Any]) -> dict[str, Any]:
    """Remove excluded fields before hashing."""
    return {k: v for k, v in obj.items() if k not in HASH_EXCLUSIONS}
```

---

## 5. Code Identity (`code_hash`)

### Base Rule

```
code_hash = blake3(git_commit_sha || dirty_patch_hash?)
```

| Component | Source | Notes |
|-----------|--------|-------|
| `git_commit_sha` | `git rev-parse HEAD` | 40-char hex |
| `dirty_patch_hash` | `blake3(git diff HEAD)` | Only if working tree is dirty |

### Dirty Detection

```python
def get_code_identity() -> CodeIdentity:
    commit_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    
    diff_output = subprocess.check_output(
        ["git", "diff", "HEAD"], text=True
    )
    
    is_dirty = len(diff_output) > 0
    dirty_patch_hash = blake3(diff_output.encode()) if is_dirty else None
    
    return CodeIdentity(
        commit_sha=commit_sha,
        is_dirty=is_dirty,
        dirty_patch_hash=dirty_patch_hash,
    )
```

### Normalization

| Rule | Enforcement |
|------|-------------|
| Line endings | **LF only** via `.gitattributes`: `* text=auto eol=lf` |
| Untracked files | **Excluded** from dirty hash (tracked files only) |
| Staged vs unstaged | Both included in `git diff HEAD` |

---

## 6. Environment Identity (`env_hash`)

### Identity-Critical Inputs

```python
ENV_IDENTITY_INPUTS: list[str] = [
    # Lock files
    "poetry.lock",      # or conda.lock / requirements.txt hash
    
    # Runtime versions
    "python_version",   # e.g., "3.12.1"
    
    # GPU stack (affects TensorRT plan determinism)
    "cuda_version",     # e.g., "12.4"
    "cudnn_version",    # e.g., "9.1.0"
    "tensorrt_version", # e.g., "10.0.1"
    
    # Compute backends (affect numerical results)
    "blas_backend",     # "mkl" | "openblas" | "accelerate"
    "blas_version",     # e.g., "2024.0.0"
    
    # XLA/JAX if used
    "xla_version",      # if applicable
]
```

### Debug Provenance (Not Hashed)

Logged but not included in `env_hash`:

| Field | Example | Rationale |
|-------|---------|-----------|
| `os_kernel` | `6.5.0-generic` | Rarely affects results; aids drift triage |
| `gpu_driver_version` | `550.54.14` | Rarely affects results |
| `gpu_model` | `NVIDIA A100` | Informational |
| `gpu_compute_capability` | `8.0` | Informational |

### Hash Computation

```python
def compute_env_hash() -> bytes:
    components = [
        get_lockfile_hash(),
        f"python={sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        f"cuda={get_cuda_version()}",
        f"cudnn={get_cudnn_version()}",
        f"tensorrt={get_tensorrt_version()}",
        f"blas={get_blas_backend()}@{get_blas_version()}",
    ]
    
    canonical = "\n".join(sorted(components))
    return blake3(canonical.encode()).digest()
```

---

## 7. Data Identity (`data_version_hash`)

### Inputs

```python
@dataclass
class DataManifest:
    """Manifest of dataset identity."""
    entries: list[DataManifestEntry]
    
@dataclass  
class DataManifestEntry:
    symbol: str
    date_range: tuple[date, date]
    vendor: str
    vendor_snapshot_id: str | None  # Immutable snapshot ID if vendor provides
    schema_version: str
```

### Hash Computation

```python
def compute_data_version_hash(manifest: DataManifest) -> bytes:
    # Sort entries for determinism
    sorted_entries = sorted(
        manifest.entries, 
        key=lambda e: (e.symbol, e.date_range[0])
    )
    
    canonical = json.dumps(
        [asdict(e) for e in sorted_entries],
        cls=CanonicalJSONEncoder,
        sort_keys=True,
    )
    
    return blake3(canonical.encode()).digest()
```

### What We Don't Hash

- **Full file checksums**: Too expensive at scale; trust manifest + vendor snapshot ID
- **Row counts**: Can change with corrections; use vendor snapshot ID instead

---

## 8. Hash Algorithm Selection

| Use Case | Algorithm | Output Size | Rationale |
|----------|-----------|-------------|-----------|
| Identity hashes (content, env, code, data) | **BLAKE3** | 32 bytes | Fast, secure, streaming |
| Fast dedup checks (non-cryptographic) | **XXH3-128** | 16 bytes | Fastest; CAS lookup only |
| Legacy compatibility | SHA-256 | 32 bytes | Only if external systems require |

### Implementation

```python
import blake3
import xxhash

def hash_for_identity(data: bytes) -> bytes:
    """Cryptographic hash for identity purposes."""
    return blake3.blake3(data).digest()

def hash_for_dedup(data: bytes) -> bytes:
    """Fast hash for deduplication checks."""
    return xxhash.xxh3_128(data).digest()
```

---

## 9. Composite Hash Construction

### Artifact Content Hash

```python
def compute_content_hash(artifact: dict[str, Any], quantization_policy: dict[str, str]) -> bytes:
    """Compute content hash for an artifact."""
    # 1. Strip excluded fields
    cleaned = strip_exclusions(artifact)
    
    # 2. Apply quantization policies to float fields
    quantized = apply_quantization(cleaned, quantization_policy)
    
    # 3. Check for banned values
    _check_banned_values(quantized)
    
    # 4. Canonicalize to JSON (RFC 8785)
    canonical = canonicalize_json(quantized)
    
    # 5. Hash
    return hash_for_identity(canonical.encode("utf-8"))
```

### Plan Hash

```python
def compute_plan_hash(
    strategy_id: str,
    hyperparams: dict[str, Any],
    feature_config: dict[str, Any],
    code_hash: bytes,
    env_hash: bytes,
    data_version_hash: bytes,
) -> bytes:
    """Compute plan hash from all identity components."""
    components = {
        "strategy_id": strategy_id,
        "hyperparams": hyperparams,
        "feature_config": feature_config,
        "code_hash": code_hash.hex(),
        "env_hash": env_hash.hex(),
        "data_version_hash": data_version_hash.hex(),
    }
    
    canonical = canonicalize_json(components)
    return hash_for_identity(canonical.encode("utf-8"))
```

---

## 10. Contract Violations

| Violation | Severity | Behavior |
|-----------|----------|----------|
| NaN/Inf in hashable field | **ERROR** | Raise `HashingContractViolation`; abort registration |
| Float without quantization policy | **ERROR** | Raise `HashingContractViolation` |
| Unknown quantization policy ID | **ERROR** | Raise `HashingContractViolation` |
| Missing required field | **ERROR** | Raise `HashingContractViolation` |
| Excluded field in hash input | **WARNING** | Log warning; strip automatically |

---

## 11. Table / Frame Identities (CanonicalFrame v1)

### 11.1 No Arrow IPC / Parquet as Canonical Identity Bytes

- **Rule:** Arrow IPC streams and Parquet files are **not** canonical identity bytes.
- **Rationale:** Dictionary encoding, metadata ordering, and writer-specific behaviors can introduce nondeterminism for the same logical table.
- **Consequence:** Hashes of Arrow/Parquet bytes may be used for *caching* or *transport*, but **must not** be used as persistent identities (CAS IDs, registry IDs, bundle manifests).

### 11.2 Reserved Namespace: `frame.v1`

We reserve a namespace for future table identities:

- `frame.v1:<algo>:<digest>`

Where:

- `frame.v1` encodes the **canonicalization spec** for tabular data, including:
  - Deterministic row ordering.
  - Column ordering and naming.
  - NaN / null semantics.
  - Timezone normalization.
- `<algo>` will typically be `b3-256` (BLAKE3-256) for identity purposes.

**Status:** Reserved only. No implementation of CanonicalFrame v1 is provided in this document; future ADRs will define the exact byte format and hashing rules.

---

## 12. Feature-Matrix Fingerprints and Row Ordering

### 12.1 Default Mode: Order-Sensitive (`fingerprint.v1:strict`)

- **Default:** Feature-matrix fingerprints are **row-order sensitive**.
- **Namespace:** `fingerprint.v1:strict:<algo>:<digest>`
- **Implications:**
  - Schema (columns, types) and row order both affect the hash.
  - Callers are responsible for ensuring deterministic ordering (e.g., by primary key and timestamp) before hashing.

### 12.2 Alternative Modes (Explicitly Namespaced)

Order-insensitive or sorted variants must use distinct namespaces, e.g.:

- `fingerprint.v1:sorted(keys=[...]):<algo>:<digest>` — canonicalizer sorts by the specified keys before hashing.
- `fingerprint.v1:unordered:<algo>:<digest>` — true order-insensitive hashing.

**Restrictions:**

- `fingerprint.v1:unordered` is **forbidden** for persistent IDs (registries, manifests, CAS IDs).
- It may be used only for **telemetry** and short-lived diagnostics, where collisions are tolerable.

---

## 13. SipHash Rotation and Cache Key Namespacing

### 13.1 Use Case

SipHash is appropriate for:

- Untrusted/external inputs used as cache keys (e.g., Redis/L3 cache).
- Protection against hash-flooding without exposing internal hash schemes.

### 13.2 Key Format with Key IDs

To support safe rotation, all SipHash-based keys must include a **key id** (`kid`) and a versioned namespace, for example:

```text
cache.v1:sip24:kid=<kid>:<digest>
```

- `cache.v1` — cache-namespace and schema version.
- `sip24` — algorithm family (e.g., SipHash-2-4).
- `kid` — key identifier in the secrets manager.
- `<digest>` — hex-encoded SipHash output.

### 13.3 Rotation Policy

- Keys are stored in a secrets manager with at least:
  - `kid_active` — currently active SipHash key.
  - `kid_prev` — previous key retained for a limited period.
- **Default reads:**
  - Use only `kid_active` to avoid double lookups.
- **Optional dual-read for expensive caches:**
  - Try `kid_active`; on miss, optionally try `kid_prev`.
  - Retain `kid_prev` for at least `2 * max_ttl` of the cache if dual-read is enabled.

---

## 14. Adaptive Hashing: Where It Is Allowed vs Forbidden

### 14.1 Forbidden Domains (No Adaptive Hashing)

Adaptive hashing (silently switching algorithms for the same logical key space) is **forbidden** for:

- CAS artifact IDs (e.g., `cas.v1:b3-256:<hex>`).
- Registry IDs and promotion identities (plan hashes, env/data identities).
- Bundle and manifest fields intended for **cross-run** or **cross-system** use (e.g., `bundle_manifest.json`, `artifact_index.json`).
- Any identifiers that are part of gate-visible or audit-visible control-plane operations.

Changing the hash algorithm for these domains requires:

- A **new namespace** (e.g., `cas.v2:...`) or
- A new **domain** (e.g., `cas.v2`, `artifact.v2`) with explicit migration semantics.

### 14.2 Allowed Domains (With Constraints)

Adaptive hashing is allowed only for:

- Telemetry-only identifiers and fingerprints where collisions are acceptable.
- Short-lived internal caches whose keys are naturally ephemeral and live in clearly versioned namespaces (e.g., `cache.v1:...`).

Even in these domains, any change to algorithm should:

- Bump a **version prefix** (e.g., `cache.v2:...`) rather than silently reusing `cache.v1`.
- Be documented in release notes or migration guides when it affects operator-facing behavior.

---

## Appendix: Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2025-01-15 | BLAKE3 as primary hash | Faster than SHA-256; equally secure; streaming support |
| 2025-01-15 | Strict float prohibition (Q_NONE default) | Eliminates cross-platform drift; decimal strings for money |
| 2025-01-15 | Scientific notation for quantized floats | Deterministic formatting across Python versions |
| 2025-01-15 | BLAS backend added to env_hash | MKL vs OpenBLAS can cause numerical differences |
| 2025-01-15 | Vendor snapshot ID over file checksums | Scale concerns; trust vendor immutability guarantees |
