import asyncio
import sys
from collections.abc import Awaitable, Collection, Coroutine, Iterator
from contextlib import suppress
from types import TracebackType
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Set,
    Type,
    TypeVar,
    Union,
)

from ._job import Job

if sys.version_info >= (3, 11):
    from asyncio import timeout as asyncio_timeout
    from typing import Self
else:
    from async_timeout import timeout as asyncio_timeout

    Self = TypeVar("Self", bound="Scheduler")

_T = TypeVar("_T")
_FutureLike = Union["asyncio.Future[_T]", Awaitable[_T]]
ExceptionHandler = Callable[["Scheduler", Dict[str, Any]], None]


class Scheduler(Collection[Job[object]]):

    __slots__ = (
        "_jobs",
        "_shields",
        "_close_timeout",
        "_wait_timeout",
        "_limit",
        "_exception_handler",
        "_failed_tasks",
        "_failed_task",
        "_pending",
        "_closed",
        "_pause_count",
    )

    def __init__(
        self,
        *,
        close_timeout: Optional[float] = 0.1,
        wait_timeout: Optional[float] = 60,
        limit: Optional[int] = 100,
        pending_limit: int = 10000,
        exception_handler: Optional[ExceptionHandler] = None,
    ):
        if exception_handler is not None and not callable(exception_handler):
            raise TypeError(
                f"A callable object or None is expected, got {exception_handler!r}"
            )

        self._jobs: Set[Job[object]] = set()
        self._shields: Set[asyncio.Task[object]] = set()
        self._close_timeout = close_timeout
        self._wait_timeout = wait_timeout
        self._limit = limit
        self._exception_handler = exception_handler
        self._failed_tasks: asyncio.Queue[Optional[asyncio.Task[object]]] = (
            asyncio.Queue()
        )
        self._failed_task: Optional[asyncio.Task[None]] = None
        if sys.version_info < (3, 10):
            self._failed_task = asyncio.create_task(self._wait_failed())
        self._pending: asyncio.Queue[Job[object]] = asyncio.Queue(maxsize=pending_limit)
        self._closed = False
        self._pause_count = 0

    def __iter__(self) -> Iterator[Job[Any]]:
        return iter(self._jobs)

    def __len__(self) -> int:
        return len(self._jobs)

    def __contains__(self, obj: object) -> bool:
        return obj in self._jobs

    def __repr__(self) -> str:
        info = []
        if self._closed:
            info.append("closed")
        if self._pause_count:
            info.append("paused")
        state = " ".join(info)
        if state:
            state += " "
        return f"<Scheduler {state}jobs={len(self)}>"

    async def __aenter__(self: Self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        await self.wait_and_close()

    @property
    def limit(self) -> Optional[int]:
        return self._limit

    @property
    def pending_limit(self) -> int:
        return self._pending.maxsize

    @property
    def close_timeout(self) -> Optional[float]:
        return self._close_timeout

    @property
    def active_count(self) -> int:
        return len(self._jobs) - self._pending.qsize()

    @property
    def pending_count(self) -> int:
        return self._pending.qsize()

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def paused(self) -> bool:
        return self._pause_count > 0

    def pause(self) -> "_PauseGuard":
        """Return an async context manager suspending jobs scheduling.

        While at least one guard is held, :meth:`spawn` still accepts new
        jobs (respecting :attr:`pending_limit`) but keeps them pending
        instead of starting them.  Guards can be nested; scheduling
        resumes when the last held guard is released.
        """
        return _PauseGuard(self)

    async def drain(self) -> None:
        """Wait for jobs that are running right now to finish.

        Jobs spawned after this call (e.g. pending ones while the
        scheduler is paused) don't extend the wait.  Returns immediately
        if nothing is running.  Cancelling the caller doesn't cancel
        the running jobs.
        """
        jobs = [job for job in self._jobs if job.active]
        if not jobs:
            return
        gather = asyncio.gather(
            *(job._wait() for job in jobs),
            return_exceptions=True,
        )
        await asyncio.shield(gather)

    def _pause_acquire(self) -> None:
        if self._closed:
            raise RuntimeError("Cannot pause a closed scheduler")
        self._pause_count += 1

    def _pause_release(self) -> None:
        assert self._pause_count > 0
        self._pause_count -= 1
        if self._pause_count == 0 and not self._closed:
            # The last guard is released, resume scheduling.
            # A closed scheduler never resumes.
            self._start_pending()

    async def spawn(
        self, coro: Coroutine[object, object, _T], name: Optional[str] = None
    ) -> Job[_T]:
        if self._closed:
            raise RuntimeError("Scheduling a new job after closing")
        if self._failed_task is None:
            self._failed_task = asyncio.create_task(self._wait_failed())
        else:
            if self._failed_task.get_loop() is not asyncio.get_running_loop():
                raise RuntimeError(f"{self!r} is bound to a different event loop")
        job = Job(coro, self, name=name)
        should_start = not self.paused and (
            self._limit is None or self.active_count < self._limit
        )
        if should_start:
            job._start()
        else:
            try:
                # wait for free slot in queue
                await self._pending.put(job)
            except asyncio.CancelledError:
                await job.close()
                raise
        self._jobs.add(job)
        return job

    def shield(self, arg: _FutureLike[_T]) -> "asyncio.Future[_T]":
        inner = asyncio.ensure_future(arg)
        if inner.done():
            return inner

        # This function is a copy of asyncio.shield(), except for the addition of
        # the below 2 lines.
        self._shields.add(inner)
        inner.add_done_callback(self._shields.discard)

        loop = inner.get_loop()
        outer = loop.create_future()

        def _inner_done_callback(inner: "asyncio.Task[object]") -> None:
            if outer.cancelled():
                if not inner.cancelled():
                    inner.exception()
                return

            if inner.cancelled():
                outer.cancel()
            else:
                exc = inner.exception()
                if exc is not None:
                    outer.set_exception(exc)
                else:
                    outer.set_result(inner.result())

        def _outer_done_callback(outer: "asyncio.Future[object]") -> None:
            if not inner.done():
                inner.remove_done_callback(_inner_done_callback)

        inner.add_done_callback(_inner_done_callback)
        outer.add_done_callback(_outer_done_callback)
        return outer

    async def wait_and_close(self, timeout: Optional[float] = None) -> None:
        if timeout is None:
            timeout = self._wait_timeout
        with suppress(asyncio.TimeoutError):
            async with asyncio_timeout(timeout):
                while self._jobs or self._shields:
                    gather = asyncio.gather(
                        *(job._wait() for job in self._jobs),
                        *self._shields,
                        return_exceptions=True,
                    )
                    await asyncio.shield(gather)
        await self.close()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True  # prevent adding new jobs

        jobs = self._jobs
        if jobs or self._shields:
            # cleanup pending queue
            # all job will be started on closing
            while not self._pending.empty():
                self._pending.get_nowait()

            for f in self._shields:
                f.cancel()

            await asyncio.gather(
                *(job._close(self._close_timeout) for job in jobs),
                *(asyncio.wait_for(f, self._close_timeout) for f in self._shields),
                return_exceptions=True,
            )
            self._jobs.clear()
        if self._failed_task is not None:
            self._failed_tasks.put_nowait(None)
            await self._failed_task

    def call_exception_handler(self, context: Dict[str, Any]) -> None:
        if self._exception_handler is None:
            asyncio.get_running_loop().call_exception_handler(context)
        else:
            self._exception_handler(self, context)

    @property
    def exception_handler(self) -> Optional[ExceptionHandler]:
        return self._exception_handler

    def _done(self, job: Job[object]) -> None:
        self._jobs.discard(job)
        self._start_pending()

    def _start_pending(self) -> None:
        if self.paused:
            return
        while self.pending_count:
            if self._limit is not None and self.active_count >= self._limit:
                return
            new_job = self._pending.get_nowait()
            if new_job.closed:
                continue
            new_job._start()

    async def _wait_failed(self) -> None:
        # a coroutine for waiting failed tasks
        # without awaiting for failed tasks async raises a warning
        while True:
            task = await self._failed_tasks.get()
            if task is None:
                return  # closing
            try:
                await task  # should raise exception
            except Exception:
                # Cleanup a warning
                # self.call_exception_handler() is already called
                # by Job._add_done_callback
                # Thus we caught an task exception and we are good citizens
                pass


class _PauseGuard:
    """An async context manager returned by :meth:`Scheduler.pause`."""

    __slots__ = ("_scheduler", "_held")

    def __init__(self, scheduler: Scheduler) -> None:
        self._scheduler = scheduler
        self._held = False

    async def __aenter__(self) -> "_PauseGuard":
        self._scheduler._pause_acquire()
        self._held = True
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        if self._held:
            self._held = False
            self._scheduler._pause_release()
