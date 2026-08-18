"""Pure task construction: builds TaskIR objects from job specs and partitions."""

from pysrc.tuning.core.tasks.dataset_views import DatasetView
from pysrc.tuning.core.tasks.labels import (
    excess_return_labels,
    forward_return_labels,
    sign_labels,
)
from pysrc.tuning.core.tasks.regime_projection import RegimeSegment, project_regime
from pysrc.tuning.core.tasks.segmentation import split_walkforward
from pysrc.tuning.core.tasks.task_builder import TaskBuilder
from pysrc.tuning.core.tasks.task_keys import make_candidate_key, make_task_key
from pysrc.tuning.core.tasks.task_manifest import TaskManifest

__all__ = [
    "TaskBuilder",
    "TaskManifest",
    "DatasetView",
    "forward_return_labels",
    "sign_labels",
    "excess_return_labels",
    "split_walkforward",
    "RegimeSegment",
    "project_regime",
    "make_task_key",
    "make_candidate_key",
]
