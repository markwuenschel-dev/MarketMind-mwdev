"""
Self-Evolving Test Infrastructure with Adaptive Intelligence
==========================================================

A test framework that learns from execution patterns and automatically
optimizes its behavior over time.
"""

import json
import statistics
import threading
import time
import warnings
from collections import defaultdict, deque
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ExecutionMetrics:
    """Detailed metrics for a single execution."""

    operation_chain: tuple[str, ...]
    data_shape: tuple[int, ...]
    execution_time_ms: float
    memory_peak_mb: float
    parallel_path_taken: bool
    cache_hit_rate: float
    error_occurred: bool
    timestamp: float = field(default_factory=time.time)


@dataclass
class OperationProfile:
    """Learned profile for an operation type."""

    name: str
    avg_execution_time_ms: float = 0.0
    execution_count: int = 0
    failure_rate: float = 0.0
    optimal_chunk_size: int | None = None
    memory_efficiency: float = 1.0  # MB per row processed
    parallel_speedup_factor: float = 1.0

    def update_timing(self, execution_time_ms: float):
        self.avg_execution_time_ms = (
            self.avg_execution_time_ms * self.execution_count + execution_time_ms
        ) / (self.execution_count + 1)
        self.execution_count += 1

    def update_failure(self, failed: bool):
        self.failure_rate = (
            self.failure_rate * self.execution_count + (1.0 if failed else 0.0)
        ) / (self.execution_count + 1)


class AdaptiveLearningEngine:
    """Core learning engine that drives system evolution."""

    def __init__(self, persistence_path: Path | None = None):
        self.persistence_path = persistence_path or Path(".test_intelligence")
        self.metrics_history: deque = deque(maxlen=1000)
        self.operation_profiles: dict[str, OperationProfile] = {}
        self.execution_patterns: dict[tuple[str, ...], list[float]] = defaultdict(list)
        self.failure_patterns: dict[str, list[str]] = defaultdict(list)
        self.parallel_effectiveness: dict[int, list[float]] = defaultdict(list)
        self._lock = threading.RLock()

        # Load existing intelligence
        self._load_intelligence()

    def record_execution(self, metrics: ExecutionMetrics):
        """Record execution metrics and trigger learning."""
        with self._lock:
            self.metrics_history.append(metrics)

            # Update operation profiles
            for op_name in metrics.operation_chain:
                if op_name not in self.operation_profiles:
                    self.operation_profiles[op_name] = OperationProfile(op_name)

                profile = self.operation_profiles[op_name]
                profile.update_timing(metrics.execution_time_ms / len(metrics.operation_chain))
                profile.update_failure(metrics.error_occurred)

            # Learn execution patterns
            self.execution_patterns[metrics.operation_chain].append(metrics.execution_time_ms)

            # Learn parallel effectiveness
            if metrics.parallel_path_taken:
                data_size = np.prod(metrics.data_shape)
                self.parallel_effectiveness[data_size].append(metrics.execution_time_ms)

            # Record failure patterns
            if metrics.error_occurred:
                pattern_key = f"{metrics.data_shape}_{metrics.operation_chain}"
                self.failure_patterns["execution_errors"].append(pattern_key)

        # Trigger periodic learning
        if len(self.metrics_history) % 50 == 0:
            self._evolve_strategies()

    def _evolve_strategies(self):
        """Core evolution logic - learns and adapts strategies."""
        with self._lock:
            self._optimize_parallelization_thresholds()
            self._optimize_operation_ordering()
            self._tune_chunk_sizes()
            self._identify_failure_patterns()

            # Persist learned intelligence
            self._save_intelligence()

    def _optimize_parallelization_thresholds(self):
        """Learn optimal data size thresholds for parallelization."""
        if len(self.parallel_effectiveness) < 10:
            return

        # Find the data size where parallel execution becomes beneficial
        sequential_times = []
        parallel_times = []

        for metrics in self.metrics_history:
            data_size = np.prod(metrics.data_shape)
            if metrics.parallel_path_taken:
                parallel_times.append((data_size, metrics.execution_time_ms))
            else:
                sequential_times.append((data_size, metrics.execution_time_ms))

        if sequential_times and parallel_times:
            # Find crossover point where parallel becomes faster
            self.learned_parallel_threshold = self._find_performance_crossover(
                sequential_times, parallel_times
            )

    def _optimize_operation_ordering(self):
        """Learn optimal ordering of operations."""
        # Track which operation sequences perform best
        operation_performance = {}

        for pattern, times in self.execution_patterns.items():
            if len(times) >= 5:  # Need sufficient samples
                avg_time = statistics.mean(times)
                operation_performance[pattern] = avg_time

        # Identify high-performing patterns
        if operation_performance:
            best_patterns = sorted(operation_performance.items(), key=lambda x: x[1])[:5]
            self.optimal_operation_sequences = [pattern for pattern, _ in best_patterns]

    def _tune_chunk_sizes(self):
        """Learn optimal chunk sizes for parallel operations."""
        for op_name, profile in self.operation_profiles.items():
            if profile.execution_count > 10:
                # Calculate memory efficiency and optimal chunk size
                recent_metrics = [m for m in self.metrics_history if op_name in m.operation_chain][
                    -10:
                ]

                if recent_metrics:
                    avg_memory = statistics.mean(m.memory_peak_mb for m in recent_metrics)
                    avg_data_size = statistics.mean(np.prod(m.data_shape) for m in recent_metrics)

                    profile.memory_efficiency = avg_memory / max(avg_data_size, 1)

                    # Adaptive chunk size based on memory efficiency
                    target_memory_mb = 100  # Target 100MB chunks
                    profile.optimal_chunk_size = int(target_memory_mb / profile.memory_efficiency)

    def _identify_failure_patterns(self):
        # Count repeated failure signatures using structured keys
        failure_analysis = defaultdict(int)
        for metrics in self.metrics_history:
            if metrics.error_occurred:
                pattern_key = (metrics.data_shape, len(metrics.operation_chain))
                failure_analysis[pattern_key] += 1
        self.high_risk_patterns = {
            pattern_key: count for pattern_key, count in failure_analysis.items() if count >= 3
        }

    def _find_performance_crossover(self, sequential_data, parallel_data):
        """Find the data size where parallel execution becomes beneficial."""
        # Simple heuristic: find where parallel average becomes lower than sequential
        sequential_by_size = defaultdict(list)
        parallel_by_size = defaultdict(list)

        for size, time_ms in sequential_data:
            size_bucket = int(size // 1000) * 1000  # Bucket into 1K groups
            sequential_by_size[size_bucket].append(time_ms)

        for size, time_ms in parallel_data:
            size_bucket = int(size // 1000) * 1000
            parallel_by_size[size_bucket].append(time_ms)

        for size_bucket in sorted(sequential_by_size.keys()):
            if size_bucket in parallel_by_size:
                seq_avg = statistics.mean(sequential_by_size[size_bucket])
                par_avg = statistics.mean(parallel_by_size[size_bucket])

                if par_avg < seq_avg * 0.9:  # 10% improvement threshold
                    return size_bucket

        return 2000  # Default fallback

    def should_parallelize(self, data_shape: tuple[int, ...], operations: list[str]) -> bool:
        # Size-based cutoff learned from prior runs
        data_size = int(np.prod(data_shape))
        thr = getattr(self, "learned_parallel_threshold", None)
        if thr is not None and data_size < thr:
            return False

        # Structured high-risk pattern guard
        pattern_key = (data_shape, len(operations))
        if getattr(self, "high_risk_patterns", None) and pattern_key in self.high_risk_patterns:
            return False

        # Heuristic: avoid parallel if any op is too failure-prone
        op_profiles = getattr(self, "operation_profiles", {})
        for op_name in operations:
            prof = op_profiles.get(op_name)
            if prof and getattr(prof, "failure_rate", 0.0) > 0.10:
                return False

        return True

    def get_optimal_chunk_size(self, operation: str, data_size: int) -> int:
        """Get learned optimal chunk size for an operation."""
        if operation in self.operation_profiles:
            profile = self.operation_profiles[operation]
            if profile.optimal_chunk_size:
                return min(profile.optimal_chunk_size, data_size)

        # Fallback heuristic
        return max(100, data_size // 8)

    def recommend_operation_order(self, operations: list[str]) -> list[str]:
        """Recommend optimal order for operations based on learned patterns."""
        if hasattr(self, "optimal_operation_sequences"):
            ops_tuple = tuple(operations)

            # Look for learned patterns
            for pattern in self.optimal_operation_sequences:
                if set(pattern) == set(ops_tuple):
                    return list(pattern)

        # Fallback: order by average execution time (fastest first)
        def get_avg_time(op):
            if op in self.operation_profiles:
                return self.operation_profiles[op].avg_execution_time_ms
            return float("inf")

        return sorted(operations, key=get_avg_time)

    def _save_intelligence(self):
        """Persist learned intelligence to disk."""
        if not self.persistence_path:
            return

        intelligence_data = {
            "operation_profiles": {
                name: {
                    "avg_execution_time_ms": profile.avg_execution_time_ms,
                    "execution_count": profile.execution_count,
                    "failure_rate": profile.failure_rate,
                    "optimal_chunk_size": profile.optimal_chunk_size,
                    "memory_efficiency": profile.memory_efficiency,
                    "parallel_speedup_factor": profile.parallel_speedup_factor,
                }
                for name, profile in self.operation_profiles.items()
            },
            "execution_patterns": {str(k): v for k, v in self.execution_patterns.items()},
            "learned_settings": getattr(self, "learned_parallel_threshold", None),
            "timestamp": time.time(),
        }

        try:
            with open(self.persistence_path, "w") as f:
                json.dump(intelligence_data, f, indent=2)
        except Exception as e:
            warnings.warn(f"Failed to save intelligence: {e}", stacklevel=2)

    def _load_intelligence(self):
        """Load previously learned intelligence."""
        if not self.persistence_path.exists():
            return

        try:
            with open(self.persistence_path) as f:
                data = json.load(f)

            # Restore operation profiles
            for name, profile_data in data.get("operation_profiles", {}).items():
                profile = OperationProfile(name)
                profile.avg_execution_time_ms = profile_data.get("avg_execution_time_ms", 0.0)
                profile.execution_count = profile_data.get("execution_count", 0)
                profile.failure_rate = profile_data.get("failure_rate", 0.0)
                profile.optimal_chunk_size = profile_data.get("optimal_chunk_size")
                profile.memory_efficiency = profile_data.get("memory_efficiency", 1.0)
                profile.parallel_speedup_factor = profile_data.get("parallel_speedup_factor", 1.0)
                self.operation_profiles[name] = profile

            # Restore execution patterns
            for pattern_str, times in data.get("execution_patterns", {}).items():
                pattern = eval(pattern_str)  # Convert string back to tuple
                self.execution_patterns[pattern] = times

            # Restore learned settings
            if "learned_settings" in data and data["learned_settings"]:
                self.learned_parallel_threshold = data["learned_settings"]

        except Exception as e:
            warnings.warn(f"Failed to load intelligence: {e}", stacklevel=2)

    def get_intelligence_report(self) -> dict[str, Any]:
        """Generate a report on learned intelligence."""
        return {
            "total_executions": len(self.metrics_history),
            "operation_profiles": {
                name: {
                    "avg_time_ms": profile.avg_execution_time_ms,
                    "executions": profile.execution_count,
                    "failure_rate": profile.failure_rate,
                    "optimal_chunk_size": profile.optimal_chunk_size,
                }
                for name, profile in self.operation_profiles.items()
            },
            "learned_parallel_threshold": getattr(self, "learned_parallel_threshold", None),
            "execution_patterns_learned": len(self.execution_patterns),
            "high_risk_patterns": getattr(self, "high_risk_patterns", {}),
        }

    def generate_scenarios_for_testing(self) -> list[dict]:
        # Snapshot under lock if present to avoid mid-update races
        if hasattr(self, "_lock"):
            self._lock.acquire()
        try:
            scenarios: list[dict] = []

            # Parallelization threshold boundary
            thr = getattr(self, "learned_parallel_threshold", None)
            if thr is not None:
                scenarios.append(
                    {
                        "kind": "parallel_threshold_minus",
                        "rows": max(thr - 1, 1),
                        "expect_parallel": False,
                    }
                )
                scenarios.append(
                    {
                        "kind": "parallel_threshold_plus",
                        "rows": thr + 1,
                        "expect_parallel": True,
                    }
                )

            # High-risk patterns: {(shape, num_ops): count}
            for (shape, num_ops), count in getattr(self, "high_risk_patterns", {}).items():
                scenarios.append(
                    {
                        "kind": "high_risk_pattern",
                        "shape": shape,
                        "num_ops": num_ops,
                        "risk_count": count,
                        "expect_stability": False,  # tests can assert: avoid parallel and ensure no crash
                    }
                )

            # Learned optimal op sequences
            for seq in getattr(self, "optimal_operation_sequences", []):
                scenarios.append(
                    {
                        "kind": "optimal_sequence",
                        "ops": list(seq),
                        "expect_reorder": True,
                    }
                )

            return scenarios
        finally:
            if hasattr(self, "_lock"):
                self._lock.release()


class SelfEvolvingProcessingEngine:
    """Processing engine enhanced with adaptive learning capabilities."""

    def __init__(self, learning_engine: AdaptiveLearningEngine):
        self.learning_engine = learning_engine
        self._operations: dict[str, Callable] = {}
        self._execution_stats = defaultdict(list)

    def register_operation(
        self, kind: str, fn: Callable, parallel_safe: bool = False, memory_intensive: bool = False
    ):
        """Register operation with learning metadata."""
        self._operations[kind] = fn

        # Initialize learning profile if needed
        if kind not in self.learning_engine.operation_profiles:
            profile = OperationProfile(kind)
            profile.memory_efficiency = 2.0 if memory_intensive else 1.0
            self.learning_engine.operation_profiles[kind] = profile

    def process(self, data: Any, spec: dict, config: Any) -> Any:
        """Process data with adaptive intelligence."""
        start_time = time.perf_counter()
        operations = spec.get("ops", [])
        operation_names = [op.get("kind", "unknown") for op in operations]

        # Get data shape for learning
        if hasattr(data, "shape"):
            data_shape = data.shape
        elif hasattr(data, "height") and hasattr(data, "width"):
            data_shape = (data.height, data.width)
        else:
            data_shape = (len(data), 1) if hasattr(data, "__len__") else (1, 1)

        # Use learned intelligence to decide execution strategy
        should_parallelize = config.parallel_enabled and self.learning_engine.should_parallelize(
            data_shape, operation_names
        )

        # Use learned operation ordering
        if len(operations) > 1:
            optimal_order = self.learning_engine.recommend_operation_order(operation_names)
            if optimal_order != operation_names:
                # Reorder operations based on learned intelligence
                operations = self._reorder_operations(operations, optimal_order)

        error_occurred = False
        parallel_path_taken = False

        try:
            if should_parallelize and not config.force_sequential:
                result = self._execute_parallel_intelligent(data, operations, config)
                parallel_path_taken = True
            else:
                result = self._execute_sequential(data, operations)

        except Exception:
            error_occurred = True
            raise

        finally:
            # Record execution for learning
            execution_time_ms = (time.perf_counter() - start_time) * 1000

            metrics = ExecutionMetrics(
                operation_chain=tuple(operation_names),
                data_shape=data_shape,
                execution_time_ms=execution_time_ms,
                memory_peak_mb=self._estimate_memory_usage(data),
                parallel_path_taken=parallel_path_taken,
                cache_hit_rate=0.0,  # Would need cache integration
                error_occurred=error_occurred,
            )

            self.learning_engine.record_execution(metrics)

        return result

    def _execute_parallel_intelligent(self, data, operations, config):
        """Execute with learned chunk sizing and executor selection."""
        if not operations:
            return data

        # Use learned chunk size for the most expensive operation
        primary_op = operations[0]["kind"]
        data_size = data.height if hasattr(data, "height") else len(data)
        optimal_chunk_size = self.learning_engine.get_optimal_chunk_size(primary_op, data_size)

        # Determine optimal number of chunks
        num_chunks = min(config.max_workers, max(1, data_size // optimal_chunk_size))

        # Use learned executor selection
        use_process_executor = (
            config.parallel_executor == "process"
            and all(op.get("picklable", False) for op in operations)
            and data_size > 10000  # Only for large datasets
        )

        Executor = ProcessPoolExecutor if use_process_executor else ThreadPoolExecutor

        # Execute with intelligent chunking
        if hasattr(data, "iloc"):  # pandas
            chunk_size = len(data) // num_chunks
            chunks = [data.iloc[i : i + chunk_size] for i in range(0, len(data), chunk_size)]
        else:  # polars
            chunk_size = max(1, data.height // num_chunks)
            starts = range(0, data.height, chunk_size)
            chunks = [data.slice(s, min(chunk_size, data.height - s)) for s in starts]

        with Executor(max_workers=num_chunks) as executor:
            registry = self._operations
            processed_chunks = list(
                executor.map(
                    _apply_ops_to_chunk,
                    chunks,
                    [operations] * len(chunks),
                    [registry] * len(chunks),
                )
            )

        # Concatenate results
        if hasattr(data, "iloc"):  # pandas
            return __import__("pandas").concat(processed_chunks).sort_index()
        else:  # polars
            return __import__("polars").concat(processed_chunks)

    def _execute_sequential(self, data, operations):
        """Execute operations sequentially."""
        for op_spec in operations:
            params = {
                k: v for k, v in op_spec.items() if k not in ("kind", "parallel_safe", "picklable")
            }
            data = self._operations[op_spec["kind"]](data, **params)
        return data

    def _reorder_operations(self, operations: list[dict], optimal_order: list[str]) -> list[dict]:
        """Reorder operations based on learned optimal sequence."""
        operation_map = {op["kind"]: op for op in operations}
        return [operation_map[op_name] for op_name in optimal_order if op_name in operation_map]

    def _estimate_memory_usage(self, data) -> float:
        """Rough estimate of memory usage in MB."""
        if hasattr(data, "memory_usage"):
            return data.memory_usage(deep=True).sum() / (1024 * 1024)
        elif hasattr(data, "estimated_size"):
            return data.estimated_size() / (1024 * 1024)
        else:
            # Rough estimate
            if hasattr(data, "__len__"):
                return len(data) * 8 / (1024 * 1024)  # 8 bytes per element estimate
            return 1.0


def _apply_ops_to_chunk(chunk, ops, registry):
    """Worker function for parallel processing (must be at module level for pickling)."""
    for op_spec in ops:
        params = {
            k: v for k, v in op_spec.items() if k not in ("kind", "parallel_safe", "picklable")
        }
        fn = registry[op_spec["kind"]]
        chunk = fn(chunk, **params)
    return chunk


# Example usage and integration
def create_intelligent_test_infrastructure():
    """Factory function to create fully integrated intelligent infrastructure."""

    learning_engine = AdaptiveLearningEngine()
    processing_engine = SelfEvolvingProcessingEngine(learning_engine)

    # Register some example operations
    def robust_scale(data, input_col: str, output_col: str = None, out_col: str = None, **kwargs):
        if out_col is not None and output_col is None:
            output_col = out_col
        output_col = output_col or f"{input_col}_robust"

        if hasattr(data, "with_columns"):  # polars
            median_val = data[input_col].median()
            mad = (data[input_col] - median_val).abs().median()
            mad = 1.0 if mad == 0 or mad is None else mad
            return data.with_columns(((data[input_col] - median_val) / mad).alias(output_col))
        else:  # pandas
            data = data.copy()
            median_val = data[input_col].median()
            mad = (data[input_col] - median_val).abs().median()
            mad = 1.0 if __import__("pandas").isna(mad) or mad == 0 else mad
            data[output_col] = (data[input_col] - median_val) / mad
            return data

    processing_engine.register_operation(
        "scaling.robust", robust_scale, parallel_safe=True, memory_intensive=False
    )

    return learning_engine, processing_engine
