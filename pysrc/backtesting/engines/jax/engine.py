from __future__ import annotations

from pysrc.backtesting.contracts.errors import OptionalDependencyMissingError
from pysrc.backtesting.contracts.registry import register_engine


class JaxEngineAdapter:
    def run(self, plan, data, store):
        raise OptionalDependencyMissingError(
            "jax engine is optional and not enabled in this scaffold. Use engine_id='vectorized.sma' or install the JAX backend before enabling it."
        )


register_engine("jax.scaffold", lambda: JaxEngineAdapter())
