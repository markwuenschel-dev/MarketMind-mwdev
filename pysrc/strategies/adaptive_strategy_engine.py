"""
Next-Gen Self-Evolving Strategy Adapter for pipeline_strategy.py

This adapter creates an autonomous strategy evolution system that:
- Monitors performance drift across multiple timeframes
- Automatically triggers reoptimization when drift is detected
- Manages ensemble of strategies with dynamic blending
- Implements online learning with rolling windows
- Provides adaptive parameter spaces that evolve over time
- Handles regime detection and strategy switching
"""

# py/strategies/adaptive_strategy_engine.py
import json
import threading
from collections import defaultdict, deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, get_type_hints

import pandas as pd

try:
    import pysrc.strategies.momentum  # noqa: F401
except ImportError:
    pass
try:
    import pysrc.strategies.stat_arb  # noqa: F401
except ImportError:
    pass

import numpy as np

try:
    import optuna  # type: ignore
except (ImportError, ModuleNotFoundError):
    optuna = None  # type: ignore[assignment]

# Import the base framework components
# Zero-arg-compatible shim so tests can call mod.DriftState() without params
from dataclasses import dataclass as _dataclass

from pysrc.strategies.pipeline_strategy import (
    BacktestConfig,
    ChampionChallenger,
    PipelineStrategy,
    StrategyContext,
    StrategyRegistry,
    TradeIntent,
    _score_nav,
    detect_drift,
    optuna_tune,
    parameter_sweep,
)


@_dataclass
class DriftState:
    ref_mean: float = 0.0
    ref_std: float = 1.0


from pysrc.ops.mm_logkit import get_logger

LOG = get_logger(__name__)


def _to_float(x: Any) -> float:
    # numpy/pandas scalar with .item()
    if hasattr(x, "item"):
        try:
            return float(x.item())
        except (TypeError, ValueError):
            pass

    # first element of a basic sequence
    if isinstance(x, (list, tuple)):
        try:
            return float(x[0])
        except (IndexError, TypeError, ValueError):
            pass

    # numpy array-like: mean as float (only if numpy is present)
    try:
        import numpy as np  # type: ignore
    except ImportError:
        np = None
    if np is not None and "numpy" in type(x).__module__:
        try:
            return float(np.asarray(x, dtype=float).mean())
        except (TypeError, ValueError, OverflowError):
            pass

    # final fallback
    return float(x)


# --------------------------------------------------------------------------------------
# Evolution Events and Callbacks
# --------------------------------------------------------------------------------------


@dataclass
class EvolutionEvent:
    timestamp: pd.Timestamp
    event_type: str  # "drift_detected", "reoptimization", "champion_changed", "blend_updated"
    strategy_name: str
    old_params: dict[str, Any]
    new_params: dict[str, Any]
    performance_delta: float
    metadata: dict[str, Any] = field(default_factory=dict)


class EvolutionCallback(Protocol):
    def on_evolution_event(self, event: EvolutionEvent) -> None: ...


# --------------------------------------------------------------------------------------
# Adaptive Parameter Space
# --------------------------------------------------------------------------------------


class AdaptiveParameterSpace:
    """Parameter space that evolves based on historical performance."""

    def __init__(
        self,
        base_space: dict[str, tuple[float, float]],
        adaptation_rate: float = 0.1,
        memory_length: int = 100,
    ):
        self.base_space = base_space
        self.adaptation_rate = adaptation_rate
        self.memory_length = memory_length
        self.performance_history: dict[str, deque] = {
            param: deque(maxlen=memory_length) for param in base_space
        }
        self.current_space = base_space.copy()

    def update(self, params: dict[str, Any], score: float) -> None:
        """Update parameter space based on performance feedback."""
        for param, value in params.items():
            if param in self.performance_history:
                self.performance_history[param].append((value, score))

    def evolve_space(self) -> dict[str, tuple[float, float]]:
        """Evolve parameter ranges based on performance patterns."""
        new_space = {}

        for param, (base_low, base_high) in self.base_space.items():
            if len(self.performance_history[param]) < 10:
                new_space[param] = (base_low, base_high)
                continue

            # Analyze performance vs parameter values
            values, scores = zip(*self.performance_history[param], strict=False)
            values, scores = np.array(values), np.array(scores)

            # Find sweet spot and adapt range
            best_idx = np.argmax(scores)
            best_value = values[best_idx]

            # Adaptive range centering around best performance
            current_range = float(base_high) - float(base_low)
            new_center = _to_float(
                (1 - self.adaptation_rate) * (base_low + base_high) / 2
                + self.adaptation_rate * best_value
            )

            # Gradually narrow range if consistently good performance
            score_variance = _to_float(
                np.var(scores[-20:]) if len(scores) >= 20 else np.var(scores)
            )
            range_scale = _to_float(max(0.3, 1.0 - score_variance * 0.1))  # Don't narrow too much

            new_low = max(base_low, new_center - current_range * range_scale / 2)
            new_high = min(base_high, new_center + current_range * range_scale / 2)

            new_space[param] = (new_low, new_high)

        self.current_space = new_space
        return new_space


# --------------------------------------------------------------------------------------
# Multi-Timeframe Drift Monitor
# --------------------------------------------------------------------------------------


@dataclass
class MultiFrameDriftState:
    short_term: DriftState | None = None  # 20 periods
    medium_term: DriftState | None = None  # 60 periods
    long_term: DriftState | None = None  # 200 periods


class MultiTimeframeDriftMonitor:
    """Monitors drift across multiple timeframes for nuanced evolution triggers."""

    def __init__(
        self,
        short_window: int = 20,
        medium_window: int = 60,
        long_window: int = 200,
        sensitivity: float = 2.5,
    ):
        self.short_window = short_window
        self.medium_window = medium_window
        self.long_window = long_window
        self.sensitivity = sensitivity
        self.state = MultiFrameDriftState()

    def check_drift(self, returns: pd.Series) -> tuple[bool, dict[str, bool]]:
        drift_signals: dict[str, bool] = {}
        _det = globals().get("detect_drift", detect_drift)

        if len(returns) >= self.short_window:
            recent = returns.tail(self.short_window)
            self.state.short_term, drift_signals["short"] = _det(
                recent, self.state.short_term, self.sensitivity
            )

        # Medium-term (tactical shifts)
        if len(returns) >= self.medium_window:
            medium = returns.tail(self.medium_window)
            _det = globals().get("detect_drift", detect_drift)
            self.state.medium_term, drift_signals["medium"] = _det(
                medium, self.state.medium_term, self.sensitivity
            )

        # Long-term (regime changes)
        if len(returns) >= self.long_window:
            long_term = returns.tail(self.long_window)
            _det = globals().get("detect_drift", detect_drift)
            self.state.long_term, drift_signals["long"] = _det(
                long_term, self.state.long_term, self.sensitivity * 1.5
            )

        return (any(drift_signals.values()) if drift_signals else False), drift_signals


# --------------------------------------------------------------------------------------
# Strategy Ensemble Manager
# --------------------------------------------------------------------------------------


class StrategyEnsemble:
    """Manages multiple strategies with dynamic blending weights."""

    def __init__(self, strategies: list[tuple[str, dict[str, Any]]], rebalance_frequency: int = 50):
        self.strategy_specs = strategies
        self.rebalance_frequency = rebalance_frequency
        self.blend_weights: dict[str, float] = {}
        self.performance_history: dict[str, list[float]] = defaultdict(list)
        self.steps_since_rebalance = 0

    def update_performance(self, strategy_name: str, score: float) -> None:
        """Update individual strategy performance."""
        self.performance_history[strategy_name].append(score)
        self.steps_since_rebalance += 1

    def should_rebalance(self) -> bool:
        """Check if ensemble weights should be rebalanced."""
        return self.steps_since_rebalance >= self.rebalance_frequency

    def rebalance_weights(self) -> dict[str, float]:
        """Recompute blend weights based on recent performance."""
        if not self.performance_history:
            n = max(1, len(self.strategy_specs))
            self.blend_weights = {name: 1.0 / n for name, _ in self.strategy_specs}
            self.steps_since_rebalance = 0
            return self.blend_weights

        new_weights: dict[str, float] = {}
        lookback = min(20, len(next(iter(self.performance_history.values()))))

        for strategy_name, _ in self.strategy_specs:
            if strategy_name in self.performance_history:
                recent_scores = self.performance_history[strategy_name][-lookback:]
                weights = np.exp(np.linspace(-1, 0, len(recent_scores)))
                weighted_score = np.average(recent_scores, weights=weights)
            else:
                weighted_score = 0.0
            new_weights[strategy_name] = max(0.01, weighted_score)

        total = sum(new_weights.values())
        new_weights = {k: v / total for k, v in new_weights.items()}

        self.blend_weights = new_weights
        self.steps_since_rebalance = 0
        return new_weights


# --------------------------------------------------------------------------------------
# Self-Evolving Adapter (Main Class)
# --------------------------------------------------------------------------------------


class SelfEvolvingAdapter:
    """
    Next-generation self-evolving strategy adapter that autonomously manages
    strategy evolution, drift detection, parameter optimization, and ensemble blending.
    """

    def __init__(
        self,
        strategies: list[
            tuple[str, str, dict[str, tuple[float, float]]]
        ],  # (name, class_name, param_space)
        ctx: StrategyContext,
        prices: pd.DataFrame,
        evolution_frequency: int = 100,  # steps between evolution checks
        optimization_trials: int = 30,
        min_history_for_evolution: int = 50,
        backtest_cfg: BacktestConfig | None = None,
        callbacks: list[EvolutionCallback] | None = None,
    ):

        self.strategies = strategies
        self.ctx = ctx
        self.prices = prices
        self.evolution_frequency = evolution_frequency
        self.optimization_trials = optimization_trials
        self.min_history = min_history_for_evolution
        self.backtest_cfg = backtest_cfg or BacktestConfig()
        self.callbacks = callbacks or []

        # Evolution state
        self.step_count = 0
        self.evolution_history: list[EvolutionEvent] = []
        self.strategy_instances: dict[str, PipelineStrategy] = {}
        self.champion_challengers: dict[str, ChampionChallenger] = {}
        self.adaptive_spaces: dict[str, AdaptiveParameterSpace] = {}
        self.drift_monitors: dict[str, MultiTimeframeDriftMonitor] = {}
        self.ensemble = StrategyEnsemble([(name, {}) for name, _, _ in strategies])

        # Performance tracking
        self.performance_buffer: dict[str, pd.Series] = {}
        self.last_weights: pd.DataFrame | None = None

        # Thread safety
        self._lock = threading.RLock()

        # Initialize components
        self._initialize_strategies()
        self._name_to_classkey = {name: cls_key for name, cls_key, _ in self.strategies}

    @staticmethod
    def _coerce_params_for_class(strategy_cls, params: Mapping[str, float | int]) -> dict[str, Any]:
        hints = get_type_hints(getattr(strategy_cls, "__init__", object))
        out: dict[str, Any] = {}
        for k, v in params.items():
            # harden common RNG keys even if hints missing or incorrect
            if k in {"random_state", "seed"} or k.endswith("_seed"):
                out[k] = int(round(float(v)))
                continue
            t = hints.get(k)
            if t is int:
                out[k] = int(round(float(v)))
            elif t is float:
                out[k] = float(v)
            else:
                out[k] = v
        return out

    def _initialize_strategies(self) -> None:
        """Initialize all strategy components."""
        for strategy_name, class_name, param_space in self.strategies:
            # Prefer the live, patched registry mapping if present
            _reg = getattr(StrategyRegistry, "_reg", None)
            if isinstance(_reg, dict) and class_name in _reg:
                strategy_cls = _reg[class_name]
            else:
                strategy_cls = StrategyRegistry.get(class_name)

            center = {k: (low + high) / 2 for k, (low, high) in param_space.items()}
            init_params = self._coerce_params_for_class(strategy_cls, center)

            self.strategy_instances[strategy_name] = strategy_cls(**init_params)

            self.champion_challengers[strategy_name] = ChampionChallenger(
                strategy_cls=strategy_cls,
                ctx=self.ctx,
                prices=self.prices,
                backtest_cfg=self.backtest_cfg,
                champion_params=init_params,  # already a dict[str, Any]
            )

            self.adaptive_spaces[strategy_name] = AdaptiveParameterSpace(
                base_space=dict(param_space),  # Mapping → dict
                adaptation_rate=0.15,
                memory_length=200,
            )

            # Setup drift monitor
            self.drift_monitors[strategy_name] = MultiTimeframeDriftMonitor()

            # Initialize performance buffer
            self.performance_buffer[strategy_name] = pd.Series(dtype=float)

    def _fire_event(self, event: EvolutionEvent) -> None:
        """Fire evolution event to all callbacks."""
        self.evolution_history.append(event)
        for callback in self.callbacks:
            try:
                callback.on_evolution_event(event)
            except (AttributeError, TypeError, ValueError, RuntimeError) as e:
                LOG.warning(f"Callback failed: {e}")

    def generate_trade_intent(self) -> TradeIntent:
        """Main entry point - generates trade intent with evolution logic."""
        with self._lock:
            self.step_count += 1

            # Generate signals from all strategies
            strategy_intents = {}
            for strategy_name in self.strategy_instances:
                try:
                    intent = self.strategy_instances[strategy_name].generate_trade_intent(self.ctx)
                    strategy_intents[strategy_name] = intent
                except (ValueError, TypeError, KeyError, AttributeError, RuntimeError) as e:
                    LOG.error(f"Strategy {strategy_name} failed: {e}")
                    continue

            # Create ensemble blend (precomputed — no re-calls)
            if strategy_intents:
                final_weights = self._blend_precomputed(strategy_intents)
                final_intent = TradeIntent(weights=final_weights, diagnostics={})
            else:
                # Fallback to zero weights
                final_intent = TradeIntent(
                    weights=pd.Series(0.0, index=self.prices.index),
                    diagnostics={"error": "All strategies failed"},
                )

            # Update performance tracking
            self._update_performance_tracking(strategy_intents, final_intent)

            # Check for evolution triggers
            if self.step_count % self.evolution_frequency == 0:
                self._evolution_step()

            return final_intent

    def _blend_precomputed(self, intents: dict[str, TradeIntent]) -> pd.DataFrame:
        acc = None
        n = max(1, len(intents))
        # refresh weights if time to rebalance

        if self.ensemble.should_rebalance():
            self.ensemble.rebalance_weights()
        for name, intent in intents.items():
            w = intent.weights
            if isinstance(w, pd.Series):
                w = w.to_frame()
            alpha = self.ensemble.blend_weights.get(name, 1.0 / n)
            w = w * alpha
            acc = w if acc is None else acc.add(w, fill_value=0.0)
        l1 = acc.abs().sum(axis=1)
        scale = (1.0 / l1).clip(upper=1.0).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        return acc.mul(scale, axis=0)

    def _update_performance_tracking(
        self, strategy_intents: dict[str, TradeIntent], final_intent: TradeIntent
    ) -> None:
        """Update performance metrics for all strategies."""
        current_time = pd.Timestamp.now()

        # Calculate returns if we have previous weights
        if self.last_weights is not None:
            returns = self.prices.pct_change().iloc[-1]
            if isinstance(returns, pd.Series) and len(returns) > 0:
                aligned = self.last_weights.iloc[-1].reindex(returns.index).fillna(0.0)
                portfolio_return = float((aligned * returns).sum())

                # Update performance buffers
                for strategy_name in strategy_intents:
                    self.performance_buffer[strategy_name] = pd.concat(
                        [
                            self.performance_buffer[strategy_name],
                            pd.Series([portfolio_return], index=[current_time]),
                        ]
                    )

                    # Update ensemble performance tracking
                    if len(self.performance_buffer[strategy_name]) > 0:
                        recent_score = _score_nav(
                            self.performance_buffer[strategy_name].cumsum() + 1.0
                        )
                        self.ensemble.update_performance(strategy_name, recent_score)

        # Store current weights for next iteration
        weights = final_intent.weights
        if isinstance(weights, pd.Series):
            weights = weights.to_frame("weight")
        self.last_weights = weights

    def _evolution_step(self) -> None:
        """Perform evolution checks and optimizations."""
        LOG.info(f"Evolution step {self.step_count // self.evolution_frequency}")

        for strategy_name in self.strategy_instances:
            self._evolve_strategy(strategy_name)

    def _evolve_strategy(self, strategy_name: str) -> None:
        """Evolve a single strategy based on drift and performance."""
        if len(self.performance_buffer[strategy_name]) < self.min_history:
            return

        # Check for drift
        returns = self.performance_buffer[strategy_name]
        drift_detected, drift_details = self.drift_monitors[strategy_name].check_drift(returns)

        if not drift_detected:
            return

        LOG.info(f"Drift detected for {strategy_name}: {drift_details}")

        # Trigger reoptimization
        old_params = dict(self.champion_challengers[strategy_name].champion_params)
        event_to_fire = None

        try:
            # Evolve parameter space
            new_space = self.adaptive_spaces[strategy_name].evolve_space()

            # Run optimization with evolved space
            cls_key = self._name_to_classkey[strategy_name]
            _reg = getattr(StrategyRegistry, "_reg", None)
            if isinstance(_reg, dict) and cls_key in _reg:
                strategy_cls = _reg[cls_key]
            else:
                strategy_cls = StrategyRegistry.get(cls_key)
            if optuna:
                results = optuna_tune(
                    strategy_cls=strategy_cls,
                    sampler_spec=new_space,
                    ctx=self.ctx,
                    prices=self.prices,
                    backtest_cfg=self.backtest_cfg,
                    n_trials=self.optimization_trials,
                )
                best_params = results[0].params if results else old_params
            else:
                # Fallback to grid search with reduced space
                grid = {k: [low, (low + high) / 2, high] for k, (low, high) in new_space.items()}
                results = parameter_sweep(
                    strategy_cls=strategy_cls,
                    param_grid=grid,
                    ctx=self.ctx,
                    prices=self.prices,
                    backtest_cfg=self.backtest_cfg,
                    n_jobs=1,
                )
                best_params = results[0].params if results else old_params

            # Coerce types to match the strategy's __init__ signature
            best_params = self._coerce_params_for_class(strategy_cls, best_params)

            # Update champion
            new_champion = self.champion_challengers[strategy_name].step(
                challenger_params=best_params,
                improvement=0.02,  # Require 2% improvement
            )

            # Final guard: if registry didn't have a mapping, keep the existing concrete type
            if strategy_cls is PipelineStrategy:
                strategy_cls = type(self.strategy_instances[strategy_name])

            self.strategy_instances[strategy_name] = strategy_cls(**new_champion)

            # Update adaptive space with new result
            if results:
                self.adaptive_spaces[strategy_name].update(new_champion, results[0].score)

            # Prepare success event
            performance_delta = results[0].score if results else 0.0
            event_to_fire = EvolutionEvent(
                timestamp=pd.Timestamp.now(),
                event_type="reoptimization",
                strategy_name=strategy_name,
                old_params=old_params,
                new_params=new_champion,
                performance_delta=performance_delta,
                metadata={"drift_details": drift_details, "trials": len(results) if results else 0},
            )

        except (ValueError, TypeError, KeyError, AttributeError, RuntimeError) as e:
            LOG.error(f"Evolution failed for {strategy_name}: {e}")
            # Still fire an event for the failed evolution attempt
            event_to_fire = EvolutionEvent(
                timestamp=pd.Timestamp.now(),
                event_type="evolution_failed",
                strategy_name=strategy_name,
                old_params=old_params,
                new_params=old_params,  # No change since evolution failed
                performance_delta=0.0,
                metadata={"drift_details": drift_details, "error": str(e)},
            )

        # Fire the event outside the try-except so callbacks always get notified
        if event_to_fire is not None:
            self._fire_event(event_to_fire)

    def get_evolution_summary(self) -> dict[str, Any]:
        """Get summary of evolution history and current state."""
        return {
            "step_count": self.step_count,
            "evolution_events": len(self.evolution_history),
            "current_blend_weights": self.ensemble.blend_weights.copy(),
            "champion_params": {
                name: cc.champion_params.copy() for name, cc in self.champion_challengers.items()
            },
            "recent_events": [
                {
                    "timestamp": e.timestamp.isoformat(),
                    "type": e.event_type,
                    "strategy": e.strategy_name,
                    "performance_delta": e.performance_delta,
                }
                for e in self.evolution_history[-10:]  # Last 10 events
            ],
        }

    def force_evolution(self, strategy_name: str | None = None) -> None:
        """Force evolution for specific strategy or all strategies."""
        with self._lock:
            if strategy_name:
                self._evolve_strategy(strategy_name)
            else:
                for name in self.strategy_instances:
                    self._evolve_strategy(name)


# --------------------------------------------------------------------------------------
# Example Evolution Callback
# --------------------------------------------------------------------------------------


class LoggingEvolutionCallback:
    """Simple callback that logs evolution events."""

    def on_evolution_event(self, event: EvolutionEvent) -> None:
        LOG.info(
            f"Evolution: {event.event_type} for {event.strategy_name} "
            f"(Δperf: {event.performance_delta:.4f})"
        )


class FileEvolutionCallback:
    """Callback that saves evolution history to JSON file."""

    def __init__(self, filepath: str | Path):
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

    def on_evolution_event(self, event: EvolutionEvent) -> None:
        event_dict = {
            "timestamp": event.timestamp.isoformat(),
            "event_type": event.event_type,
            "strategy_name": event.strategy_name,
            "old_params": event.old_params,
            "new_params": event.new_params,
            "performance_delta": event.performance_delta,
            "metadata": event.metadata,
        }

        # Append to file
        events = []
        if self.filepath.exists():
            with open(self.filepath) as f:
                events = json.load(f)

        events.append(event_dict)

        with open(self.filepath, "w") as f:
            json.dump(events, f, indent=2)


# --------------------------------------------------------------------------------------
# Usage Example
# --------------------------------------------------------------------------------------


def create_evolving_system(price_data: pd.DataFrame) -> SelfEvolvingAdapter:
    """Example of how to set up the self-evolving system."""

    # Define strategy ensemble with parameter spaces
    strategies = [
        (
            "momentum_1",
            "momentum",
            {
                "random_state": (1000, 9999),
            },
        ),
        (
            "momentum_2",
            "momentum",
            {
                "random_state": (1000, 9999),
            },
        ),
        # Could add more strategies like stat_arb_pairs, etc.
    ]

    # Setup context
    ctx = StrategyContext(
        prices=price_data, backend="pandas", cache_dir=".cache_evolving", random_state=42
    )

    # Setup callbacks
    callbacks = [LoggingEvolutionCallback(), FileEvolutionCallback("evolution_log.json")]

    # Create evolving adapter
    adapter = SelfEvolvingAdapter(
        strategies=strategies,
        ctx=ctx,
        prices=price_data,
        evolution_frequency=50,  # Check every 50 steps
        optimization_trials=20,
        callbacks=callbacks,
    )

    return adapter


# Example usage:
# adapter = create_evolving_system(your_price_data)
# for i in range(1000):  # Simulate 1000 trading steps
#     intent = adapter.generate_trade_intent()
#     # Execute trades, update prices, etc.
#     summary = adapter.get_evolution_summary()
#     if i % 100 == 0:
#         print(f"Step {i}: {summary['evolution_events']} evolution events")
