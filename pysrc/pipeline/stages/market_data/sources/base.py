"""Deprecated compatibility imports for source adapters.

New code should import ``DataSource`` from
``pysrc.pipeline.stages.market_data.sources.contracts`` and ``APIDataSource``
from ``pysrc.pipeline.stages.market_data.sources.runtime``.
"""

from pysrc.pipeline.stages.market_data.sources.contracts import DataSource
from pysrc.pipeline.stages.market_data.sources.runtime import APIDataSource

__all__ = ["APIDataSource", "DataSource"]
