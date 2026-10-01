import asyncio
import heapq
import itertools
import sys
from collections.abc import Awaitable, Collection, Coroutine, Iterator
from contextlib import suppress
from types import TracebackType
from typing import (
    Any,
    Callable,
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
        "_exception_handler",
        "_failed_tasks",
        "_failed_task",
        "_pending",
        "_pending_slot",
        "_delayed",
        "_delayed_jobs",
        "_delayed_handle",
        "_delayed_seq",
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
        self._exception_handler = exception_handler
        self._failed_tasks: asyncio.Queue[Optional[asyncio.Task[object]]] = (
            asyncio.Queue()
        )
        self._failed_task: Optional[asyncio.Task[None]] = None
        if sys.version_info < (3, 10):
            self._failed_task = asyncio.create_task(self._wait_failed())
        self._pending: asyncio.Queue[Job[object]] = asyncio.Queue(maxsize=pending_limit)
        self._pending_slot = asyncio.Event()
        self._delayed: List[Tuple[float, int, Job[object]]] = []
        self._delayed_jobs: Set[Job[object]] = set()
        self._delayed_handle: Optional[asyncio.TimerHandle] = None
        self._delayed_seq = itertools.count()
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
        return len(self._jobs) - self.pending_count

    @property
    def pending_count(self) -> int:
        return self._pending.qsize() + len(self._delayed_jobs)

    @property
    def closed(self) -> bool:
        return self._closed

    async def spawn(
        self, coro: Coroutine[object, object, _T], name: Optional[str] = None
    ) -> Job[_T]:
        self._check_accepting_jobs()
        job = Job(coro, self, name=name)
        await self._admit(job)
        self._jobs.add(job)
        return job

    async def spawn_at(
        self,
        when: float,
        coro: Coroutine[object, object, _T],
        name: Optional[str] = None,
    ) -> Job[_T]:
        """Spawn a new job that starts not earlier than *when*.

        *when* is an absolute timestamp on the running event loop's
        monotonic clock (the same scale as ``loop.time()``).

        The job does not occupy ``active_count`` until the deadline is
        reached, but it is accounted in ``pending_count`` and respects
        ``pending_limit``.  Once due, the job still respects the
        concurrency ``limit``.  Jobs with equal deadlines are started
        in submission order.
        """
        self._check_accepting_jobs()
        job = Job(coro, self, name=name)
        if when <= self._monotonic():
            # the deadline has already passed, admit the job as usual
            await self._admit(job)
        else:
            try:
                # wait for free slot in pending capacity
                await self._acquire_pending_capacity()
            except (asyncio.CancelledError, RuntimeError):
                await job.close()
                raise
            self._delayed_jobs.add(job)
            heapq.heappush(self._delayed, (when, next(self._delayed_seq), job))
            self._schedule_delayed_wakeup()
        self._jobs.add(job)
        return job

    def _check_accepting_jobs(self) -> None:
        if self._closed:
            raise RuntimeError("Scheduling a new job after closing")
        if self._failed_task is None:
            self._failed_task = asyncio.create_task(self._wait_failed())
        else:
            if self._failed_task.get_loop() is not asyncio.get_running_loop():
                raise RuntimeError(f"{self!r} is bound to a different event loop")

    async def _admit(self, job: Job[object]) -> None:
        should_start = self._limit is None or self.active_count < self._limit
        if should_start:
            job._start()
        else:
            try:
                # wait for free slot in queue
                await self._acquire_pending_capacity()
                self._pending.put_nowait(job)
            except (asyncio.CancelledError, RuntimeError):
                await job.close()
                raise

    async def _acquire_pending_capacity(self) -> None:
        maxsize = self._pending.maxsize
        if maxsize <= 0:
            return  # infinite pending capacity
        while self.pending_count >= maxsize:
            if self._closed:
                raise RuntimeError("Scheduling a new job after closing")
            self._pending_slot.clear()
            if self.pending_count < maxsize or self._closed:
                continue
            await self._pending_slot.wait()

    def _monotonic(self) -> float:
        return asyncio.get_running_loop().time()

    def _call_at(self, when: float, callback: Callable[[], None]) -> asyncio.TimerHandle:
        return asyncio.get_running_loop().call_at(when, callback)

    def _schedule_delayed_wakeup(self) -> None:
        # drop stale heap entries left by jobs closed before their deadline
        while self._delayed and self._delayed[0][2] not in self._delayed_jobs:
            heapq.heappop(self._delayed)
        if not self._delayed:
            if self._delayed_handle is not None:
                self._delayed_handle.cancel()
                self._delayed_handle = None
            return
        when = self._delayed[0][0]
        handle = self._delayed_handle
        if handle is not None:
            if handle.when() == when:
                return
            # a closer deadline was inserted, move the wakeup earlier
            handle.cancel()
        self._delayed_handle = self._call_at(when, self._start_due_delayed)

    def _start_due_delayed(self) -> None:
        self._delayed_handle = None
        now = self._monotonic()
        while self._delayed and self._delayed[0][0] <= now:
            _, _, job = heapq.heappop(self._delayed)
            if job not in self._delayed_jobs:
                continue  # stale entry, the job was closed before its deadline
            if job.closed:
                self._delayed_jobs.discard(job)
                continue
            # check the limit before discarding the job from the delayed set
            # so the job doesn't count itself as active
            should_start = self._limit is None or self.active_count < self._limit
            self._delayed_jobs.discard(job)
            if should_start:
                job._start()
            else:
                # the concurrency limit is still in effect for due jobs;
                # pending capacity was reserved on submission
                self._pending.put_nowait(job)
        self._schedule_delayed_wakeup()
        self._pending_slot.set()

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

        # stop the delayed-job machinery; delayed jobs themselves are in
        # self._jobs and are closed below without waiting for their deadline
        if self._delayed_handle is not None:
            self._delayed_handle.cancel()
            self._delayed_handle = None
        self._delayed.clear()
        self._delayed_jobs.clear()
        self._pending_slot.set()  # wake up spawners waiting for pending capacity

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
        if job in self._delayed_jobs:
            # a delayed job was closed before its deadline
            self._delayed_jobs.discard(job)
            self._pending_slot.set()
        if self._pending.empty():
            return
        # No pending jobs in the queue when limit is None
        # Safe to subtract.
        ntodo = self._limit - self.active_count  # type: ignore[operator]
        i = 0
        while i < ntodo:
            if self._pending.empty():
                return
            new_job = self._pending.get_nowait()
            self._pending_slot.set()
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
