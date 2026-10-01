import asyncio
import sys
from collections import deque
from collections.abc import Awaitable, Collection, Coroutine, Iterator
from contextlib import suppress
from contextvars import Context
from types import TracebackType
from typing import (
    Any,
    Callable,
    Deque,
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
        "_failed_task_loop",
        "_pending",
        "_pending_putters",
        "_pending_limit",
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
        self._failed_tasks: Optional[
            asyncio.Queue[Optional[asyncio.Task[object]]]
        ] = None
        self._failed_task: Optional[asyncio.Task[None]] = None
        self._failed_task_loop: Optional[asyncio.AbstractEventLoop] = None
        if sys.version_info < (3, 10):
            loop = asyncio.get_running_loop()
            self._failed_tasks = asyncio.Queue()
            self._failed_task = Context().run(
                loop.create_task, self._wait_failed()
            )
            self._failed_task_loop = loop
        self._pending: Deque[Job[object]] = deque()
        self._pending_putters: Deque[
            Tuple[asyncio.Future[None], Job[object]]
        ] = deque()
        self._pending_limit = pending_limit
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
        return len(self._jobs) - len(self._pending)

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def closed(self) -> bool:
        return self._closed

    async def spawn(
        self, coro: Coroutine[object, object, _T], name: Optional[str] = None
    ) -> Job[_T]:
        if self._closed:
            raise RuntimeError("Scheduling a new job after closing")
        loop = asyncio.get_running_loop()
        if self._failed_task is None:
            self._failed_tasks = asyncio.Queue()
            self._failed_task = Context().run(loop.create_task, self._wait_failed())
            self._failed_task_loop = loop
        elif self._failed_task_loop is not loop:
            raise RuntimeError(f"{self!r} is bound to a different event loop")
        job = Job(coro, self, name=name)
        should_start = self._limit is None or self.active_count < self._limit
        if should_start:
            self._jobs.add(job)
            job._start()
        elif self._has_pending_capacity():
            self._jobs.add(job)
            self._pending.append(job)
        else:
            future = asyncio.get_running_loop().create_future()
            entry = (future, job)
            self._pending_putters.append(entry)
            try:
                await future
            except asyncio.CancelledError:
                if entry in self._pending_putters:
                    self._pending_putters.remove(entry)
                    job._reject()
                raise
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

        while self._pending_putters:
            future, job = self._pending_putters.popleft()
            if future.cancelled():
                job._reject()
            elif not future.done():
                job._reject()
                future.set_exception(RuntimeError("Scheduler is closing"))

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
            assert self._failed_tasks is not None
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
        if self._closed:
            return
        self._start_pending()
        self._admit_pending_putters()

    def _discard_pending(self, job: Job[object]) -> None:
        try:
            self._pending.remove(job)
        except ValueError:
            pass

    def _has_pending_capacity(self) -> bool:
        return self._pending_limit <= 0 or len(self._pending) < self._pending_limit

    def _start_pending(self) -> None:
        if self._limit is None:
            return
        while self._pending and self.active_count < self._limit:
            new_job = self._pending.popleft()
            if not new_job.closed:
                new_job._start()

    def _admit_pending_putters(self) -> None:
        if self._closed:
            return
        while self._has_pending_capacity() and self._pending_putters:
            future, new_job = self._pending_putters.popleft()
            if future.done():
                if future.cancelled():
                    new_job._reject()
                continue
            self._jobs.add(new_job)
            self._pending.append(new_job)
            future.set_result(None)
        self._start_pending()

    async def _wait_failed(self) -> None:
        assert self._failed_tasks is not None
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
