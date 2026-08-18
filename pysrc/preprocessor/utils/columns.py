from __future__ import annotations

import warnings

from pysrc.preprocessor.ops.common.columns import *  # noqa: F403

warnings.warn(
    "pysrc.preprocessor.utils.columns is deprecated; use pysrc.preprocessor.ops.common.columns",
    DeprecationWarning,
    stacklevel=2,
)
