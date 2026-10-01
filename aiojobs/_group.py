import asyncio
from collections.abc import Collection, Coroutine, Iterator
from types import TracebackType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Type, TypeVar

from ._job import Job

if TYPE_CHECKING:
    from ._scheduler import Scheduler

_T = TypeVar("_T")


class GroupError(Exception):
    """One or more jobs of a group have failed.

    The *failures* attribute maps every failed job to the exception
    it has finished with, so the original failure is never masked by
    errors raised by other members while the group is being cancelled.
    """

    def __init__(self, failures: Dict[Job[Any], BaseException]) -> None:
        self.failures: Dict[Job[Any], BaseException] = failures
        details = "; ".join(
            f"{job!r} failed with {exc!r}" for job, exc in failures.items()
        )
        super().__init__(f"{len(failures)} job(s) failed: {details}")


class Group(Collection[Job[Any]]):
    """A group of jobs spawned through the same scheduler.

    The group inherits the concurrency and pending limits of its
    scheduler.  The first regular exception raised by a member cancels
    the other running members and discards the ones that have not
    started yet; ``wait()`` reports every failed job afterwards.
    """

    __slots__ = ("_scheduler", "_jobs", "_wrapped", "_closed", "_aborting")

    def __init__(self, scheduler: "Scheduler") -> None:
        self._scheduler = scheduler
        self._jobs: Set[Job[Any]] = set()
        self._wrapped: Dict[Job[Any], Coroutine[object, object, Any]] = {}
        self._closed = False
        self._aborting = False

    def __iter__(self) -> Iterator[Job[Any]]:
        return iter(self._jobs)

    def __len__(self) -> int:
        return len(self._jobs)

    def __contains__(self, obj: object) -> bool:
        return obj in self._jobs

    def __repr__(self) -> str:
        state = "closed " if self._closed else ""
        return f"<Group {state}jobs={len(self)}>"

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def scheduler(self) -> "Scheduler":
        return self._scheduler

    async def __aenter__(self) -> "Group":
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        if exc_type is None:
            try:
                await self.wait()
            finally:
                self._closed = True
        else:
            # Don't shadow the exception raised by the body with
            # errors from the cancelled members.
            await self.close()

    async def spawn(
        self, coro: Coroutine[object, object, _T], name: Optional[str] = None
    ) -> Job[_T]:
        if self._closed:
            raise RuntimeError("Scheduling a new job after group is closed")
        holder: List[Job[_T]] = []

        async def runner() -> _T:
            job = holder[0]
            self._wrapped.pop(job, None)
            try:
                return await coro
            except asyncio.CancelledError:
                raise
            except Exception:
                # The first regular failure aborts the whole group.
                self._abort(job)
                raise

        job = await self._scheduler.spawn(runner(), name=name)
        # The group retrieves and reports failures of its own jobs,
        # so the scheduler must not report them as well.
        job._explicit = True
        holder.append(job)
        self._wrapped[job] = coro
        self._jobs.add(job)
        return job

    async def wait(self) -> None:
        """Wait for all members and raise GroupError if any of them failed."""
        failures: Dict[Job[Any], BaseException] = {}
        for job in list(self._jobs):
            try:
                await job.wait()
            except asyncio.CancelledError:
                task = job._task
                if task is None or not task.cancelled():
                    # The wait itself was cancelled, not the job.
                    raise
                # A cancelled member is not a failure.
            except Exception as exc:
                failures[job] = exc
        if failures:
            raise GroupError(failures)

    async def close(self) -> None:
        """Close the group, cancelling all unfinished members.

        Errors raised by members while they are being cancelled are
        retrieved but swallowed, so they cannot mask the failure that
        triggered the closing nor leak as unretrieved exceptions.
        """
        if self._closed:
            return
        self._closed = True
        self._abort(None)
        await asyncio.gather(
            *(job._close(self._scheduler.close_timeout) for job in self._jobs),
            return_exceptions=True,
        )

    def _abort(self, source: Optional[Job[Any]]) -> None:
        if self._aborting:
            return
        self._aborting = True
        for job in self._jobs:
            if job is source or job.closed:
                continue
            task = job._task
            if task is None:
                # Discard a pending member: mark it closed so the
                # scheduler never starts it, then start and cancel the
                # task at once, so the wrapper is never executed.
                job._closed = True
                job._start()
                task = job._task
            task.cancel()
            # If the wrapper has not started yet it never will, so the
            # wrapped coroutine has to be closed explicitly to avoid
            # a "coroutine was never awaited" warning.
            coro = self._wrapped.pop(job, None)
            if coro is not None:
                coro.close()
