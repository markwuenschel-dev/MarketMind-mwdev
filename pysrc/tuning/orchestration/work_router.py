"""WorkRouter: routes task IRs to the appropriate executor based on resource tags."""

from __future__ import annotations


class WorkRouter:
    """Routes tasks to local, multiprocessing, GPU, or distributed executors."""

    def route(self, task_id: str, resource_tags: dict[str, str]) -> str:
        """Return the executor name for a task given its resource requirements."""
        if resource_tags.get("device") == "gpu":
            return "gpu_batch"
        if resource_tags.get("distributed") == "true":
            return "distributed"
        if resource_tags.get("n_workers", "1") != "1":
            return "multiprocessing"
        return "local"


__all__ = ["WorkRouter"]
