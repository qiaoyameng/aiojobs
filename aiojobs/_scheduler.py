import asyncio
import heapq
import sys
from collections import deque
from collections.abc import Awaitable, Collection, Coroutine, Iterator
from contextlib import suppress
from types import TracebackType
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    List,
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


class Scheduler(Collection[Job[object]]):

    __slots__ = (
        "_jobs",
        "_shields",
        "_close_timeout",
        "_wait_timeout",
        "_limit",
        "_pending_limit",
        "_exception_handler",
        "_failed_tasks",
        "_failed_task",
        "_pending",
        "_scheduled",
        "_scheduled_count",
        "_pending_value",
        "_pending_waiters",
        "_wake_handle",
        "_wake_time",
        "_wake_generation",
        "_sequence",
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
        self._pending_limit = pending_limit
        self._exception_handler = exception_handler
        self._failed_tasks: asyncio.Queue[Optional[asyncio.Task[object]]] = (
            asyncio.Queue()
        )
        self._failed_task: Optional[asyncio.Task[None]] = None
        if sys.version_info < (3, 10):
            self._failed_task = asyncio.create_task(self._wait_failed())
        self._pending: Deque[Job[object]] = deque()
        self._scheduled: List[Tuple[float, int, Job[object]]] = []
        self._scheduled_count = 0
        self._pending_waiters: Deque["asyncio.Future[None]"] = deque()
        if pending_limit <= 0:
            self._pending_value: Optional[int] = None
        else:
            self._pending_value = pending_limit
        self._wake_handle: Optional[asyncio.TimerHandle] = None
        self._wake_time: Optional[float] = None
        self._wake_generation = 0
        self._sequence = 0
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
        return self._pending_limit

    @property
    def close_timeout(self) -> Optional[float]:
        return self._close_timeout

    @property
    def active_count(self) -> int:
        return len(self._jobs) - self.pending_count

    @property
    def pending_count(self) -> int:
        return len(self._pending) + self._scheduled_count

    @property
    def closed(self) -> bool:
        return self._closed

    async def spawn(
        self, coro: Coroutine[object, object, _T], name: Optional[str] = None
    ) -> Job[_T]:
        return await self.spawn_at(coro, None, name=name)

    async def spawn_at(
        self,
        coro: Coroutine[object, object, _T],
        start_time: Optional[float],
        *,
        name: Optional[str] = None,
    ) -> Job[_T]:
        if self._closed:
            raise RuntimeError("Scheduling a new job after closing")
        if self._failed_task is None:
            self._failed_task = asyncio.create_task(self._wait_failed())
        else:
            if self._failed_task.get_loop() is not asyncio.get_running_loop():
                raise RuntimeError(f"{self!r} is bound to a different event loop")

        job: Optional[Job[_T]] = None
        sequence = self._sequence
        self._sequence += 1
        try:
            if start_time is not None and start_time > self._time():
                await self._acquire_pending_slot()
                job = Job(coro, self, name=name, sequence=sequence)
                job._has_pending_slot = True
                self._jobs.add(job)
                self._add_scheduled(job, start_time)
            elif self._limit is None or self.active_count < self._limit:
                job = Job(coro, self, name=name, sequence=sequence)
                self._jobs.add(job)
                job._start()
            else:
                await self._acquire_pending_slot()
                job = Job(coro, self, name=name, sequence=sequence)
                job._has_pending_slot = True
                self._jobs.add(job)
                if self._limit is None or self.active_count < self._limit:
                    self._start_job(job)
                else:
                    self._pending.append(job)
        except asyncio.CancelledError:
            if job is None:
                coro.close()
            else:
                await job.close()
            raise
        except RuntimeError:
            if job is not None:
                await job.close()
            else:
                coro.close()
            raise
        return job

    async def _acquire_pending_slot(self) -> None:
        loop = asyncio.get_running_loop()
        while self._pending_value == 0:
            waiter = loop.create_future()
            self._pending_waiters.append(waiter)
            try:
                await waiter
            except asyncio.CancelledError:
                self._pending_waiters.remove(waiter)
                raise

        if self._closed:
            raise RuntimeError("Scheduling a new job after closing")
        assert self._pending_value is not None
        self._pending_value -= 1

    def _time(self) -> float:
        return asyncio.get_running_loop().time()

    def _call_at(
        self, when: float, callback: "Callable[[], None]"
    ) -> asyncio.TimerHandle:
        return asyncio.get_running_loop().call_at(when, callback)

    def _add_scheduled(self, job: Job[object], start_time: float) -> None:
        heapq.heappush(self._scheduled, (start_time, job._sequence, job))
        self._scheduled_count += 1
        if self._wake_time is None or start_time < self._wake_time:
            self._schedule_wake(start_time)

    def _schedule_wake(self, when: float) -> None:
        generation = self._wake_generation
        if self._wake_handle is not None:
            self._wake_handle.cancel()

        def wake() -> None:
            if generation != self._wake_generation:
                return
            self._wake()

        self._wake_handle = self._call_at(when, wake)
        self._wake_time = when

    def _cancel_wake(self) -> None:
        self._wake_generation += 1
        if self._wake_handle is not None:
            self._wake_handle.cancel()
            self._wake_handle = None
        self._wake_time = None

    def _wake(self) -> None:
        if self._closed:
            return

        self._wake_generation += 1
        self._wake_handle = None
        self._wake_time = None

        now = self._time()
        while self._scheduled and self._scheduled[0][0] <= now:
            start_time, _, job = heapq.heappop(self._scheduled)
            self._scheduled_count -= 1
            if job.closed:
                self._release_pending_slot(job)
                continue

            if self._limit is None or self.active_count < self._limit:
                self._start_job(job)
            else:
                self._pending.append(job)

        if self._scheduled:
            self._schedule_wake(self._scheduled[0][0])

    def _start_job(self, job: Job[object]) -> None:
        self._release_pending_slot(job)
        job._start()

    def _release_pending_slot(self, job: Job[object]) -> None:
        if job._has_pending_slot:
            job._has_pending_slot = False
            self._release_pending_waiter()

    def _release_pending_waiter(self) -> None:
        if self._pending_value is None:
            return
        self._pending_value += 1
        while self._pending_waiters:
            waiter = self._pending_waiters.popleft()
            if not waiter.done():
                waiter.set_result(None)
                return

    def _wake_all_pending_waiters(self) -> None:
        while self._pending_waiters:
            waiter = self._pending_waiters.popleft()
            if not waiter.done():
                waiter.set_result(None)

    def _discard_pending(self, job: Job[object]) -> None:
        try:
            self._pending.remove(job)
        except ValueError:
            pass
        else:
            self._release_pending_slot(job)
            return

        try:
            index = next(
                index for index, entry in enumerate(self._scheduled) if entry[2] is job
            )
        except StopIteration:
            return

        last_entry = self._scheduled.pop()
        if index < len(self._scheduled):
            self._scheduled[index] = last_entry
            heapq.heapify(self._scheduled)
        self._scheduled_count -= 1
        self._release_pending_slot(job)
        self._cancel_wake()
        if self._scheduled:
            self._schedule_wake(self._scheduled[0][0])

    async def _cancel_scheduled_jobs(self) -> None:
        jobs = []
        while self._scheduled:
            _, _, job = heapq.heappop(self._scheduled)
            self._scheduled_count -= 1
            if not job.closed:
                jobs.append(job)
        self._cancel_wake()
        await asyncio.gather(
            *(job._close(self._close_timeout) for job in jobs),
            return_exceptions=True,
        )

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
                    await self._cancel_scheduled_jobs()
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

        self._wake_all_pending_waiters()
        await self._cancel_scheduled_jobs()

        jobs = self._jobs
        if jobs or self._shields:
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
        if not self.pending_count:
            return
        # No pending jobs when limit is None
        # Safe to subtract.
        ntodo = self._limit - self.active_count  # type: ignore[operator]
        i = 0
        while i < ntodo:
            if not self._pending:
                return
            new_job = self._pending.popleft()
            if new_job.closed:
                self._release_pending_slot(new_job)
                continue
            self._start_job(new_job)
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
