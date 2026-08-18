from __future__ import annotations

import warnings

from pysrc.preprocessor.domain.market_calendar import *  # noqa: F403

warnings.warn(
    "pysrc.preprocessor.utils.market_time is deprecated; use pysrc.preprocessor.domain.market_calendar",
    DeprecationWarning,
    stacklevel=2,
)
