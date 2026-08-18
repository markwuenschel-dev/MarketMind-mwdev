"""JobStateMachine: REGISTERING → RUNNING → COMPLETE / FAILED state transitions."""

from __future__ import annotations

from enum import StrEnum


class JobState(StrEnum):
    REGISTERING = "REGISTERING"
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


_VALID_TRANSITIONS: dict[JobState, set[JobState]] = {
    JobState.REGISTERING: {JobState.RUNNING, JobState.FAILED},
    JobState.RUNNING: {JobState.COMPLETE, JobState.FAILED, JobState.CANCELLED},
    JobState.COMPLETE: set(),
    JobState.FAILED: set(),
    JobState.CANCELLED: set(),
}


class InvalidTransitionError(RuntimeError):
    """Raised when an illegal state transition is attempted."""


class JobStateMachine:
    """Tracks job state and enforces valid transitions."""

    def __init__(self, job_id: str) -> None:
        self.job_id = job_id
        self._state = JobState.REGISTERING

    @property
    def state(self) -> JobState:
        return self._state

    def transition(self, to: JobState) -> None:
        """Advance to *to*; raises InvalidTransitionError if the transition is illegal."""
        allowed = _VALID_TRANSITIONS.get(self._state, set())
        if to not in allowed:
            raise InvalidTransitionError(
                f"Job '{self.job_id}': cannot transition {self._state} → {to}. Allowed: {allowed}"
            )
        self._state = to


__all__ = ["JobState", "JobStateMachine", "InvalidTransitionError"]
