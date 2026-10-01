import asyncio
from collections.abc import Coroutine
from typing import TYPE_CHECKING, Dict, Generic, List, Optional, Set, Tuple, TypeVar

from ._job import Job

_T = TypeVar("_T")
_R = TypeVar("_R")

if TYPE_CHECKING:
    from ._scheduler import Scheduler


class JobGroupError(Exception):
    """Aggregated failures from one or more jobs in a group."""

    def __init__(self, errors: List[Tuple[Job[object], BaseException]]) -> None:
        self._errors = tuple(errors)
        details = "\n".join(
            f"  {index}. {job.get_name() or job!r}: "
            f"{type(exc).__name__}: {exc}"
            for index, (job, exc) in enumerate(self._errors, start=1)
        )
        super().__init__(f"Job group failed with {len(errors)} error(s):\n{details}")

    @property
    def errors(self) -> Tuple[Tuple[Job[object], BaseException], ...]:
        return self._errors


class JobGroup(Generic[_T]):
    __slots__ = (
        "_scheduler",
        "_pending",
        "_members",
        "_results",
        "_errors",
        "_closed",
        "_failed",
        "_closing",
        "_waiters",
        "_spawners",
        "_cleanup_task",
        "_error",
    )

    def __init__(self, scheduler: "Scheduler") -> None:
        self._scheduler = scheduler
        self._pending: Set[Job[object]] = set()
        self._members: List[Job[object]] = []
        self._results: Dict[Job[object], object] = {}
        self._errors: Dict[Job[object], BaseException] = {}
        self._closed = False
        self._failed = False
        self._closing = asyncio.get_running_loop().create_future()
        self._waiters: Set[asyncio.Future[None]] = set()
        self._spawners: Set[asyncio.Task[None]] = set()
        self._cleanup_task: Optional[asyncio.Task[None]] = None
        self._error: Optional[JobGroupError] = None
        scheduler._groups.add(self)

    @property
    def closed(self) -> bool:
        return self._closed

    async def __aenter__(self) -> "JobGroup[_T]":
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        if exc_type is None:
            await self.wait()
        else:
            await self.close()

    async def spawn(
        self, coro: "Coroutine[object, object, _R]", name: Optional[str] = None
    ) -> Job[_R]:
        if self._closed:
            coro.close()
            raise RuntimeError("Scheduling a new job after closing group")

        job = Job(coro, self._scheduler, name=name, group=self)
        self._members.append(job)
        self._pending.add(job)

        async def schedule() -> None:
            await self._scheduler._spawn_job(job)

        spawner = asyncio.create_task(schedule())
        self._spawners.add(spawner)

        def spawner_done(task: asyncio.Task[None]) -> None:
            self._spawners.discard(task)
            if not task.cancelled():
                task.exception()

        spawner.add_done_callback(spawner_done)

        try:
            await spawner
        except BaseException:
            self._pending.discard(job)
            self._remove_member(job)
            if job._task is None and not job.closed:
                job._discard_pending()
            raise
        return job

    async def wait(self) -> List[object]:
        if not self._closed and self._pending:
            waiter = asyncio.get_running_loop().create_future()
            self._waiters.add(waiter)
            try:
                await waiter
            except asyncio.CancelledError:
                self._waiters.discard(waiter)
                raise

        if self._cleanup_task is not None:
            await self._cleanup_task
            if self._error is not None:
                raise self._error

        if not self._closed:
            self._closed = True
            if not self._closing.done():
                self._closing.set_result(None)
            self._scheduler._groups.discard(self)

        if self._failed:
            raise self._error

        if self._error is not None:
            raise self._error

        return [self._results[job] for job in self._members]

    async def close(self) -> None:
        if not self._closed:
            self._begin_close()
        if self._cleanup_task is not None:
            await self._cleanup_task
        if self._error is not None:
            raise self._error

    def _fail(self) -> None:
        self._failed = True
        self._begin_close()

    def _begin_close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if not self._closing.done():
            self._closing.set_result(None)

        for job in list(self._pending):
            if job.closed:
                continue
            if job.pending:
                self._scheduler._discard_pending(job)
                job._discard_pending()
            elif job._task is not None:
                job._task.cancel()

        for spawner in self._spawners:
            spawner.cancel()

        self._cleanup_task = asyncio.create_task(self._cleanup())

    async def _cleanup(self) -> None:
        jobs = [
            job
            for job in tuple(self._members)
            if not job.closed and job._task is not None
        ]
        if self._spawners:
            await asyncio.gather(*self._spawners, return_exceptions=True)
        if jobs:
            await asyncio.gather(
                *(self._close_job(job) for job in jobs),
                return_exceptions=True,
            )

        if self._errors:
            self._error = JobGroupError(list(self._errors.items()))
        self._scheduler._groups.discard(self)
        self._resolve_waiters()

    async def _close_job(self, job: Job[object]) -> None:
        try:
            await job._close(self._scheduler.close_timeout)
        except asyncio.CancelledError:
            task = job._task
            if task is None or not task.done():
                raise
            try:
                exc = task.exception()
            except asyncio.CancelledError:
                return
            if exc is not None and not isinstance(exc, asyncio.CancelledError):
                self._errors[job] = exc
            return
        except Exception as exc:
            if self._errors.get(job) is exc:
                return
            self._errors[job] = exc

    def _job_done(self, job: Job[object]) -> None:
        task = job._task
        assert task is not None

        try:
            exc = task.exception()
        except asyncio.CancelledError:
            exc = None
            cancelled = True
        else:
            cancelled = False

        if exc is not None:
            self._errors[job] = exc
        elif not cancelled:
            self._results[job] = task.result()

        self._pending.discard(job)

        if exc is not None:
            if not self._failed:
                self._fail()
        elif not self._failed and not self._pending:
            self._resolve_waiters()

    def _job_discarded(self, job: Job[object]) -> None:
        self._pending.discard(job)
        self._remove_member(job)
        self._results.pop(job, None)
        if not self._failed and not self._pending:
            self._resolve_waiters()

    def _remove_member(self, job: Job[object]) -> None:
        try:
            self._members.remove(job)
        except ValueError:
            pass

    def _resolve_waiters(self) -> None:
        waiters = self._waiters
        self._waiters = set()
        for waiter in waiters:
            if not waiter.done():
                waiter.set_result(None)
