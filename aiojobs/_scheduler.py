import asyncio
import heapq
import math
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
    Tuple,
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
_PendingJobEntry = Tuple[int, int, "Job[object]"]


class _PriorityQueue(asyncio.PriorityQueue[Job[object]]):
    def __init__(self, maxsize: int, aging_enabled: bool) -> None:
        self._aging_enabled = aging_enabled
        super().__init__(maxsize=maxsize)

    def _entry(self, job: Job[object], now: float) -> _PendingJobEntry:
        return (-job._effective_priority(now), job._sequence, job)

    def _put(self, item: Job[object]) -> None:
        now = asyncio.get_running_loop().time()
        item._queued_at = now
        heapq.heappush(self._queue, self._entry(item, now))

    def _get(self) -> Job[object]:
        if self._aging_enabled and self._queue:
            now = asyncio.get_running_loop().time()
            self._queue = [
                self._entry(job, now) for _, _, job in self._queue
            ]
            heapq.heapify(self._queue)
        _, _, job = heapq.heappop(self._queue)
        job._queued_at = None
        return job

    def priority_changed(self, job: Job[object]) -> None:
        for index, (_, _, queued_job) in enumerate(self._queue):
            if queued_job is job:
                now = asyncio.get_running_loop().time()
                self._queue[index] = self._entry(job, now)
                heapq.heapify(self._queue)
                return

    def discard(self, job: Job[object]) -> bool:
        for index, (_, _, queued_job) in enumerate(self._queue):
            if queued_job is job:
                removed = self._queue.pop(index)
                heapq.heapify(self._queue)
                self._wakeup_next(self._putters)
                removed[2]._queued_at = None
                return True
        return False

    def clear(self) -> None:
        for _, _, job in self._queue:
            job._queued_at = None
        self._queue.clear()
        while self._putters:
            putter = self._putters.popleft()
            if not putter.done():
                putter.cancel()


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
        "_next_sequence",
        "_aging_interval",
        "_aging_step",
        "_closed",
    )

    def __init__(
        self,
        *,
        close_timeout: Optional[float] = 0.1,
        wait_timeout: Optional[float] = 60,
        limit: Optional[int] = 100,
        pending_limit: int = 10000,
        exception_handler: Optional[ExceptionHandler] = None,
        aging_interval: Optional[float] = None,
        aging_step: int = 1,
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
        self._next_sequence = 0
        if aging_interval is not None and (
            isinstance(aging_interval, bool)
            or not isinstance(aging_interval, (int, float))
        ):
            raise TypeError("aging_interval must be a number or None")
        if aging_interval is not None:
            if not math.isfinite(aging_interval) or aging_interval <= 0:
                raise ValueError("aging_interval must be a positive number")
        if isinstance(aging_step, bool) or not isinstance(aging_step, int):
            raise TypeError("aging_step must be an integer")
        if aging_step <= 0:
            raise ValueError("aging_step must be a positive integer")
        self._aging_interval = aging_interval
        self._aging_step = aging_step
        self._pending: _PriorityQueue = _PriorityQueue(
            maxsize=pending_limit,
            aging_enabled=aging_interval is not None,
        )
        self._closed = False

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
    def aging_interval(self) -> Optional[float]:
        return self._aging_interval

    @property
    def aging_step(self) -> int:
        return self._aging_step

    @property
    def closed(self) -> bool:
        return self._closed

    async def spawn(
        self,
        coro: Coroutine[object, object, _T],
        name: Optional[str] = None,
        *,
        priority: int = 0,
    ) -> Job[_T]:
        if self._closed:
            raise RuntimeError("Scheduling a new job after closing")
        if isinstance(priority, bool) or not isinstance(priority, int):
            raise TypeError("priority must be an integer")
        if self._failed_task is None:
            self._failed_task = asyncio.create_task(self._wait_failed())
        else:
            if self._failed_task.get_loop() is not asyncio.get_running_loop():
                raise RuntimeError(f"{self!r} is bound to a different event loop")
        job = Job(
            coro,
            self,
            name=name,
            priority=priority,
            sequence=self._next_sequence,
        )
        self._next_sequence += 1
        should_start = self._limit is None or self.active_count < self._limit
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

    def _priority_changed(self, job: Job[object]) -> None:
        self._pending.priority_changed(job)

    def _discard_pending(self, job: Job[object]) -> None:
        if self._pending.discard(job):
            self._jobs.discard(job)

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
            self._pending.clear()

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
        if self._closed:
            self._jobs.discard(job)
            return
        self._jobs.discard(job)
        if not self.pending_count:
            return
        # No pending jobs when limit is None
        # Safe to subtract.
        ntodo = self._limit - self.active_count  # type: ignore[operator]
        i = 0
        while i < ntodo:
            if not self.pending_count:
                return
            new_job = self._pending.get_nowait()
            if new_job.closed:
                continue
            new_job._start()
            i += 1

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
