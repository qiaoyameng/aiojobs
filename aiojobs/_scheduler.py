import asyncio
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


class _PendingEntry:
    """A single not-yet-started job waiting for a free slot."""

    __slots__ = ("job", "seq", "enqueued_at")

    def __init__(self, job: "Job[object]", seq: int, enqueued_at: float) -> None:
        self.job = job
        self.seq = seq
        # Monotonic timestamp of the moment the job entered the pending
        # queue; the aging bonus is computed from this point on.
        self.enqueued_at = enqueued_at


class _PendingQueue:
    """A bounded queue of pending jobs ordered by effective priority.

    The effective priority of a job is its base priority plus an aging
    bonus: for every *aging_interval* seconds spent in the queue the job
    gains one extra priority point (aging is disabled when
    *aging_interval* is ``None``).  Jobs with equal effective priority
    are dequeued in submission order, so the default configuration
    (all priorities equal, no aging) is a plain FIFO.
    """

    __slots__ = ("_maxsize", "_aging_interval", "_entries", "_seq", "_not_full")

    def __init__(self, maxsize: int, aging_interval: Optional[float]) -> None:
        self._maxsize = maxsize
        self._aging_interval = aging_interval
        self._entries: List[_PendingEntry] = []
        self._seq = itertools.count()
        # Set while the queue has room for more entries.
        self._not_full = asyncio.Event()
        self._not_full.set()

    @property
    def maxsize(self) -> int:
        return self._maxsize

    def __len__(self) -> int:
        return len(self._entries)

    def _effective_priority(self, entry: _PendingEntry, now: float) -> float:
        priority = entry.job.priority
        if self._aging_interval is None:
            return float(priority)
        waited = now - entry.enqueued_at
        return priority + waited / self._aging_interval

    async def put(self, job: "Job[object]") -> None:
        # maxsize <= 0 means an infinite queue, mirroring asyncio.Queue.
        while 0 < self._maxsize <= len(self._entries):
            self._not_full.clear()
            await self._not_full.wait()
        enqueued_at = asyncio.get_running_loop().time()
        self._entries.append(_PendingEntry(job, next(self._seq), enqueued_at))

    def pop_next(self) -> Optional["Job[object]"]:
        """Pop the pending job with the highest effective priority.

        Entries of jobs cancelled while waiting are discarded lazily
        here (they keep occupying a pending slot until this point).
        Returns ``None`` when no live pending job remains.
        """
        now = asyncio.get_running_loop().time()
        best_idx = -1
        best_key = (0.0, 0)
        idx = 0
        while idx < len(self._entries):
            entry = self._entries[idx]
            if entry.job.closed:
                del self._entries[idx]
                continue
            # Negative seq: for equal effective priority the earliest
            # submitted job wins.
            key = (self._effective_priority(entry, now), -entry.seq)
            if best_idx < 0 or key > best_key:
                best_key = key
                best_idx = idx
            idx += 1
        if best_idx < 0:
            self._not_full.set()
            return None
        entry = self._entries.pop(best_idx)
        if not 0 < self._maxsize <= len(self._entries):
            self._not_full.set()
        return entry.job

    def clear(self) -> None:
        self._entries.clear()
        self._not_full.set()


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
    )

    def __init__(
        self,
        *,
        close_timeout: Optional[float] = 0.1,
        wait_timeout: Optional[float] = 60,
        limit: Optional[int] = 100,
        pending_limit: int = 10000,
        aging_interval: Optional[float] = None,
        exception_handler: Optional[ExceptionHandler] = None,
    ):
        if exception_handler is not None and not callable(exception_handler):
            raise TypeError(
                f"A callable object or None is expected, got {exception_handler!r}"
            )
        if aging_interval is not None and aging_interval <= 0:
            raise ValueError(
                f"aging_interval must be a positive number or None, "
                f"got {aging_interval!r}"
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
        self._pending = _PendingQueue(pending_limit, aging_interval)
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
        return len(self._jobs) - len(self._pending)

    @property
    def pending_count(self) -> int:
        return len(self._pending)

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
        if self._failed_task is None:
            self._failed_task = asyncio.create_task(self._wait_failed())
        else:
            if self._failed_task.get_loop() is not asyncio.get_running_loop():
                raise RuntimeError(f"{self!r} is bound to a different event loop")
        job = Job(coro, self, name=name, priority=priority)
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
        self._jobs.discard(job)
        if not self.pending_count:
            return
        # No pending jobs when limit is None
        # Safe to subtract.
        ntodo = self._limit - self.active_count  # type: ignore[operator]
        i = 0
        while i < ntodo:
            new_job = self._pending.pop_next()
            if new_job is None:
                return
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
