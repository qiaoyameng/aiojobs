import asyncio
import sys
from collections.abc import Awaitable
from contextvars import ContextVar
from typing import Callable, List, NoReturn, Optional
from unittest import mock

import pytest

from aiojobs import Job, Scheduler

if sys.version_info >= (3, 11):
    from asyncio import timeout as asyncio_timeout
else:
    from async_timeout import timeout as asyncio_timeout

_MakeScheduler = Callable[..., Awaitable[Scheduler]]


def test_ctor(scheduler: Scheduler) -> None:
    assert len(scheduler) == 0


@pytest.mark.skipif(sys.version_info < (3, 10), reason="Requires Python 3.10+")
def test_ctor_without_loop() -> None:
    scheduler = Scheduler()
    assert len(scheduler) == 0


async def test_spawn(scheduler: Scheduler) -> None:
    async def coro() -> None:
        await asyncio.sleep(1)

    job = await scheduler.spawn(coro())
    assert not job.closed

    assert len(scheduler) == 1
    assert list(scheduler) == [job]
    assert job in scheduler


@pytest.mark.skipif(sys.version_info < (3, 10), reason="Requires Python 3.10+")
async def test_spawn_non_bound_loop() -> None:
    loop = asyncio.get_running_loop()

    async def coro() -> None:
        await asyncio.sleep(1)

    scheduler = await asyncio.to_thread(Scheduler)
    assert scheduler._failed_task is None

    job = await scheduler.spawn(coro())
    assert not job.closed

    ft = scheduler._failed_task
    assert ft is not None
    scheduler_loop = ft.get_loop()  # type: ignore[unreachable]
    assert scheduler_loop is loop

    assert len(scheduler) == 1
    assert list(scheduler) == [job]
    assert job in scheduler

    await scheduler.close()


@pytest.mark.skipif(sys.version_info < (3, 10), reason="Requires Python 3.10+")
def test_spawn_with_different_loop() -> None:
    async def func() -> None:
        await asyncio.sleep(1)

    scheduler = Scheduler()

    async def spawn1() -> Job[None]:
        job = await scheduler.spawn(func())
        assert not job.closed

        assert len(scheduler) == 1
        assert list(scheduler) == [job]
        assert job in scheduler

        return job

    async def spawn2() -> None:
        coro = func()
        with pytest.raises(RuntimeError, match=" is bound to a different event loop"):
            await scheduler.spawn(coro)

        await coro  # suppress a warning about non-awaited coroutine

        assert len(scheduler) == 1
        assert list(scheduler) == [job]

    loop1 = asyncio.new_event_loop()
    job = loop1.run_until_complete(spawn1())

    loop2 = asyncio.new_event_loop()
    loop2.run_until_complete(spawn2())

    loop2.close()
    loop1.run_until_complete(scheduler.close())
    loop1.close()


async def test_run_retval(scheduler: Scheduler) -> None:
    async def coro() -> int:
        return 1

    job = await scheduler.spawn(coro())
    ret = await job.wait()
    assert ret == 1

    assert job.closed

    assert len(scheduler) == 0
    assert list(scheduler) == []
    assert job not in scheduler


async def test_exception_in_explicit_waiting(make_scheduler: _MakeScheduler) -> None:
    exc_handler = mock.Mock()
    scheduler = await make_scheduler(exception_handler=exc_handler)

    async def coro() -> NoReturn:
        await asyncio.sleep(0)
        raise RuntimeError()

    job = await scheduler.spawn(coro())

    with pytest.raises(RuntimeError):
        await job.wait()

    assert job.closed

    assert len(scheduler) == 0
    assert list(scheduler) == []
    assert job not in scheduler
    assert not exc_handler.called


async def test_exception_non_waited_job(make_scheduler: _MakeScheduler) -> None:
    exc_handler = mock.Mock()
    scheduler = await make_scheduler(exception_handler=exc_handler)
    exc = RuntimeError()

    async def coro() -> NoReturn:
        await asyncio.sleep(0)
        raise exc

    await scheduler.spawn(coro())
    assert len(scheduler) == 1

    await asyncio.sleep(0.05)

    assert len(scheduler) == 0

    expect = {"exception": exc, "job": mock.ANY, "message": "Job processing failed"}
    if asyncio.get_running_loop().get_debug():
        expect["source_traceback"] = mock.ANY
    exc_handler.assert_called_with(scheduler, expect)


async def test_exception_on_close(make_scheduler: _MakeScheduler) -> None:
    exc_handler = mock.Mock()
    scheduler = await make_scheduler(exception_handler=exc_handler)
    exc = RuntimeError()

    fut: asyncio.Future[None] = asyncio.Future()

    async def coro() -> NoReturn:
        fut.set_result(None)
        raise exc

    await scheduler.spawn(coro())
    assert len(scheduler) == 1

    await scheduler.close()

    assert len(scheduler) == 0

    expect = {"exception": exc, "job": mock.ANY, "message": "Job processing failed"}
    if asyncio.get_running_loop().get_debug():
        expect["source_traceback"] = mock.ANY
    exc_handler.assert_called_with(scheduler, expect)


async def test_close_timeout(make_scheduler: _MakeScheduler) -> None:
    s1 = await make_scheduler()
    assert s1.close_timeout == 0.1
    s2 = await make_scheduler(close_timeout=1)
    assert s2.close_timeout == 1


async def test_scheduler_repr(scheduler: Scheduler) -> None:
    async def coro() -> None:
        await asyncio.sleep(1)

    assert repr(scheduler) == "<Scheduler jobs=0>"

    await scheduler.spawn(coro())
    assert repr(scheduler) == "<Scheduler jobs=1>"

    await scheduler.close()
    assert repr(scheduler) == "<Scheduler closed jobs=0>"


async def test_close_jobs(scheduler: Scheduler) -> None:
    async def coro() -> None:
        await asyncio.sleep(1)

    assert not scheduler.closed

    job = await scheduler.spawn(coro())
    await scheduler.close()
    assert job.closed
    assert scheduler.closed
    assert len(scheduler) == 0  # type: ignore[unreachable]
    assert scheduler.active_count == 0
    assert scheduler.pending_count == 0


async def test_exception_handler_api(make_scheduler: _MakeScheduler) -> None:
    s1 = await make_scheduler()
    assert s1.exception_handler is None

    handler = mock.Mock()
    s2 = await make_scheduler(exception_handler=handler)
    assert s2.exception_handler is handler

    with pytest.raises(TypeError):
        await make_scheduler(exception_handler=1)

    s3 = await make_scheduler(exception_handler=None)
    assert s3.exception_handler is None


async def test_exception_handler_default(scheduler: Scheduler) -> None:
    handler = mock.Mock()
    event_loop = asyncio.get_running_loop()
    event_loop.set_exception_handler(handler)
    d = {"a": "b"}
    scheduler.call_exception_handler(d)
    handler.assert_called_with(event_loop, d)


async def test_wait_with_timeout(scheduler: Scheduler) -> None:
    async def coro() -> None:
        await asyncio.sleep(1)

    job = await scheduler.spawn(coro())
    with pytest.raises(asyncio.TimeoutError):
        await job.wait(timeout=0.01)
    assert job.closed
    assert len(scheduler) == 0


async def test_timeout_on_closing(make_scheduler: _MakeScheduler) -> None:
    exc_handler = mock.Mock()
    scheduler = await make_scheduler(exception_handler=exc_handler, close_timeout=0.01)
    fut1: asyncio.Future[None] = asyncio.Future()
    fut2: asyncio.Future[None] = asyncio.Future()

    async def coro() -> None:
        try:
            await fut1
        except asyncio.CancelledError:
            await fut2

    job = await scheduler.spawn(coro())
    await asyncio.sleep(0.001)
    await scheduler.close()
    assert job.closed
    assert fut1.cancelled()
    expect = {"message": "Job closing timed out", "job": job, "exception": mock.ANY}
    if asyncio.get_running_loop().get_debug():
        expect["source_traceback"] = mock.ANY
    exc_handler.assert_called_with(scheduler, expect)


async def test_exception_on_closing(make_scheduler: _MakeScheduler) -> None:
    exc_handler = mock.Mock()
    scheduler = await make_scheduler(exception_handler=exc_handler)
    fut: asyncio.Future[None] = asyncio.Future()
    exc = RuntimeError()

    async def coro() -> NoReturn:
        fut.set_result(None)
        raise exc

    job = await scheduler.spawn(coro())
    await fut
    await scheduler.close()
    assert job.closed
    expect = {"message": "Job processing failed", "job": job, "exception": exc}
    if asyncio.get_running_loop().get_debug():
        expect["source_traceback"] = mock.ANY
    exc_handler.assert_called_with(scheduler, expect)


async def test_limit(make_scheduler: _MakeScheduler) -> None:
    s1 = await make_scheduler()
    assert s1.limit == 100
    s2 = await make_scheduler(limit=2)
    assert s2.limit == 2


async def test_pending_limit(make_scheduler: _MakeScheduler) -> None:
    s1 = await make_scheduler()
    assert s1.pending_limit == 10000
    s2 = await make_scheduler(pending_limit=2)
    assert s2.pending_limit == 2


async def test_pending_queue_infinite(make_scheduler: _MakeScheduler) -> None:
    scheduler = await make_scheduler(limit=1)

    async def coro(fut: "asyncio.Future[None]") -> None:
        await fut

    fut1: asyncio.Future[None] = asyncio.Future()
    fut2: asyncio.Future[None] = asyncio.Future()
    fut3: asyncio.Future[None] = asyncio.Future()

    await scheduler.spawn(coro(fut1))
    assert scheduler.pending_count == 0

    await scheduler.spawn(coro(fut2))
    assert scheduler.pending_count == 1

    await scheduler.spawn(coro(fut3))
    assert scheduler.pending_count == 2


async def test_pending_queue_limit_wait(make_scheduler: _MakeScheduler) -> None:
    scheduler = await make_scheduler(limit=1, pending_limit=1)

    async def coro(fut: "asyncio.Future[None]") -> None:
        await asyncio.sleep(0)
        await fut

    fut1: asyncio.Future[None] = asyncio.Future()
    fut2: asyncio.Future[None] = asyncio.Future()
    fut3: asyncio.Future[None] = asyncio.Future()

    await scheduler.spawn(coro(fut1))
    assert scheduler.active_count == 1
    assert scheduler.pending_count == 0

    await scheduler.spawn(coro(fut2))
    assert scheduler.active_count == 1
    assert scheduler.pending_count == 1

    with pytest.raises(asyncio.TimeoutError):
        # try to wait for 1 sec to add task to pending queue
        async with asyncio_timeout(1):
            await scheduler.spawn(coro(fut3))

    assert scheduler.active_count == 1
    assert scheduler.pending_count == 1


async def test_spawn_context_is_copied_at_submission(
    make_scheduler: _MakeScheduler,
) -> None:
    request_id: ContextVar[str] = ContextVar("request_id")
    scheduler = await make_scheduler(limit=1)

    first_started = asyncio.Event()
    release_first = asyncio.Event()
    first_result: asyncio.Future[str] = asyncio.get_running_loop().create_future()
    second_result: asyncio.Future[str] = asyncio.get_running_loop().create_future()

    async def first_worker() -> None:
        first_started.set()
        await release_first.wait()
        first_result.set_result(request_id.get())

    async def second_worker() -> None:
        second_result.set_result(request_id.get())

    async def submit(
        value: str, worker: Callable[[], Awaitable[None]]
    ) -> Job[None]:
        request_id.set(value)
        return await scheduler.spawn(worker())

    await submit("one", first_worker)
    await first_started.wait()

    second = await submit("two", second_worker)
    assert second.pending
    request_id.set("changed after second submission")

    release_first.set()
    await second.wait()

    assert second_result.result() == "two"
    assert first_result.result() == "one"


async def test_immediate_spawn_context_is_frozen(
    make_scheduler: _MakeScheduler,
) -> None:
    request_id: ContextVar[str] = ContextVar("request_id")
    scheduler = await make_scheduler()
    result: asyncio.Future[str] = asyncio.get_running_loop().create_future()
    release = asyncio.Event()

    async def worker() -> None:
        await release.wait()
        result.set_result(request_id.get())

    request_id.set("one")
    job = await scheduler.spawn(worker())
    request_id.set("changed after submission")
    release.set()
    await job.wait()

    assert result.result() == "one"


async def test_nested_spawn_uses_submitting_job_context(
    make_scheduler: _MakeScheduler,
) -> None:
    request_id: ContextVar[str] = ContextVar("request_id")
    scheduler = await make_scheduler(limit=2, pending_limit=1)

    blocker_started = asyncio.Event()
    release_blocker = asyncio.Event()
    inner_started = asyncio.Event()
    inner_result: asyncio.Future[str] = asyncio.get_running_loop().create_future()

    async def blocker() -> None:
        blocker_started.set()
        await release_blocker.wait()

    async def inner() -> None:
        inner_result.set_result(request_id.get())
        inner_started.set()

    async def parent() -> None:
        request_id.set("parent")
        inner_job = await scheduler.spawn(inner())
        request_id.set("changed after inner submission")
        await inner_started.wait()
        await inner_job.wait()

    await scheduler.spawn(blocker())
    await blocker_started.wait()

    parent_job = await scheduler.spawn(parent())
    request_id.set("main")
    release_blocker.set()
    await parent_job.wait()
    await scheduler.close()

    assert inner_result.result() == "parent"


async def test_cancelled_waiting_spawn_rejects_its_job(
    make_scheduler: _MakeScheduler,
) -> None:
    request_id: ContextVar[str] = ContextVar("request_id")
    scheduler = await make_scheduler(limit=1, pending_limit=1)

    release_first = asyncio.Event()
    first_started = asyncio.Event()
    registered = asyncio.Event()
    rejected_job: Optional[Job[None]] = None

    class PendingPutters(type(scheduler._pending_putters)):
        def append(self, item: tuple[asyncio.Future[None], Job[None]]) -> None:
            nonlocal rejected_job
            rejected_job = item[1]
            super().append(item)
            registered.set()

    scheduler._pending_putters = PendingPutters()

    async def first() -> None:
        first_started.set()
        await release_first.wait()

    async def queued() -> None:
        pass

    async def waiting_submit(value: str) -> None:
        request_id.set(value)
        await scheduler.spawn(queued())

    await scheduler.spawn(first())
    await first_started.wait()
    await scheduler.spawn(queued())

    waiting_task = asyncio.ensure_future(waiting_submit("waiting"))
    await registered.wait()
    waiting_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting_task

    assert scheduler.active_count == 1
    assert scheduler.pending_count == 1
    release_first.set()
    await scheduler.close()
    assert not scheduler._pending_putters
    assert rejected_job is not None
    assert rejected_job.closed
    assert rejected_job._coro is None
    assert rejected_job._context is None


async def test_closing_scheduler_rejects_waiting_spawns(
    make_scheduler: _MakeScheduler,
) -> None:
    scheduler = await make_scheduler(limit=1, pending_limit=1)

    first_started = asyncio.Event()
    registered = asyncio.Event()

    class PendingPutters(type(scheduler._pending_putters)):
        def append(self, item: object) -> None:
            super().append(item)
            registered.set()

    scheduler._pending_putters = PendingPutters()

    async def first() -> None:
        first_started.set()
        await asyncio.Event().wait()

    async def queued() -> None:
        pass

    async def waiting_submit() -> None:
        await scheduler.spawn(queued())

    await scheduler.spawn(first())
    await first_started.wait()
    await scheduler.spawn(queued())
    waiting_task = asyncio.ensure_future(waiting_submit())
    await registered.wait()

    closer = asyncio.ensure_future(scheduler.close())
    with pytest.raises(RuntimeError, match="Scheduler is closing"):
        await waiting_task
    await closer


async def test_closing_pending_job_releases_context(
    make_scheduler: _MakeScheduler,
) -> None:
    request_id: ContextVar[str] = ContextVar("request_id")
    scheduler = await make_scheduler(limit=1)
    first_started = asyncio.Event()

    async def first() -> None:
        first_started.set()
        await asyncio.Event().wait()

    async def pending() -> None:
        pytest.fail("pending job was started")

    request_id.set("one")
    await scheduler.spawn(first())
    await first_started.wait()

    request_id.set("two")
    job = await scheduler.spawn(pending())
    await job.close()

    assert job._coro is None
    assert job._context is None
    await scheduler.close()


async def test_exception_handler_identifies_failed_job(
    make_scheduler: _MakeScheduler,
) -> None:
    exc_handler = mock.Mock()
    scheduler = await make_scheduler(
        limit=1,
        close_timeout=1,
        exception_handler=exc_handler,
    )
    first_failed = asyncio.Event()
    second_started = asyncio.Event()
    exc1 = RuntimeError("one")
    exc2 = RuntimeError("two")

    async def fail_first() -> NoReturn:
        first_failed.set()
        raise exc1

    async def fail_second() -> NoReturn:
        second_started.set()
        raise exc2

    job1 = await scheduler.spawn(fail_first())
    job2 = await scheduler.spawn(fail_second())

    await first_failed.wait()
    await second_started.wait()
    await scheduler.close()

    jobs = [call.args[1]["job"] for call in exc_handler.call_args_list]
    exceptions = [call.args[1]["exception"] for call in exc_handler.call_args_list]
    assert jobs == [job1, job2]
    assert exceptions == [exc1, exc2]


async def test_scheduler_concurrency_pending_limit(
    make_scheduler: _MakeScheduler,
) -> None:
    scheduler = await make_scheduler(limit=1, pending_limit=1)

    async def coro(fut: "asyncio.Future[object]") -> None:
        await fut

    futures: List[asyncio.Future[object]] = [asyncio.Future() for _ in range(3)]
    jobs = []

    async def spawn() -> None:
        for fut in futures:
            jobs.append(await scheduler.spawn(coro(fut)))

    task = asyncio.create_task(spawn())
    await asyncio.sleep(0)

    assert len(scheduler) == 2
    assert scheduler.active_count == 1
    assert scheduler.pending_count == 1

    for fut in futures:
        fut.set_result(None)

    for job in jobs:
        await job.wait()

    await task

    assert len(scheduler) == 0
    assert scheduler.active_count == 0
    assert scheduler.pending_count == 0
    assert all(job.closed for job in jobs)


async def test_scheduler_concurrency_limit(make_scheduler: _MakeScheduler) -> None:
    scheduler = await make_scheduler(limit=1)

    async def coro(fut: "asyncio.Future[None]") -> None:
        await fut

    assert scheduler.active_count == 0
    assert scheduler.pending_count == 0

    fut1: asyncio.Future[None] = asyncio.Future()
    job1 = await scheduler.spawn(coro(fut1))

    assert scheduler.active_count == 1
    assert scheduler.pending_count == 0
    assert job1.active

    fut2: asyncio.Future[None] = asyncio.Future()
    job2 = await scheduler.spawn(coro(fut2))

    assert scheduler.active_count == 1
    assert scheduler.pending_count == 1
    assert job2.pending

    fut1.set_result(None)
    await job1.wait()

    assert scheduler.active_count == 1
    assert scheduler.pending_count == 0
    assert job1.closed
    assert job2.active

    fut2.set_result(None)
    await job2.wait()

    assert scheduler.active_count == 0
    assert scheduler.pending_count == 0
    assert job1.closed
    assert job2.closed


async def test_resume_closed_task(make_scheduler: _MakeScheduler) -> None:
    scheduler = await make_scheduler(limit=1)

    async def coro(fut: "asyncio.Future[None]") -> None:
        await fut

    assert scheduler.active_count == 0

    fut1: asyncio.Future[None] = asyncio.Future()
    job1 = await scheduler.spawn(coro(fut1))

    assert scheduler.active_count == 1

    fut2: asyncio.Future[None] = asyncio.Future()
    job2 = await scheduler.spawn(coro(fut2))

    assert scheduler.active_count == 1

    await job2.close()
    assert job2.closed
    assert not job2.pending

    fut1.set_result(None)
    await job1.wait()

    assert scheduler.active_count == 0
    assert len(scheduler) == 0


async def test_concurrency_disabled(make_scheduler: _MakeScheduler) -> None:
    fut1: asyncio.Future[None] = asyncio.Future()
    fut2: asyncio.Future[None] = asyncio.Future()

    scheduler = await make_scheduler(limit=None)

    async def coro() -> None:
        fut1.set_result(None)
        await fut2

    job = await scheduler.spawn(coro())
    await fut1
    assert scheduler.active_count == 1

    fut2.set_result(None)
    await job.wait()
    assert scheduler.active_count == 0


async def test_run_after_close(scheduler: Scheduler) -> None:
    async def f() -> None:
        pass

    await scheduler.close()

    coro = f()
    with pytest.raises(RuntimeError):
        await scheduler.spawn(coro)

    with pytest.warns(RuntimeWarning):
        del coro


async def test_shield(scheduler: Scheduler) -> None:
    async def coro() -> str:
        await asyncio.sleep(0)
        return "TEST"

    result = await scheduler.shield(coro())
    assert result == "TEST"
    assert len(scheduler._shields) == 0


async def test_shielded_task_continues(scheduler: Scheduler) -> None:
    completed = False

    async def inner() -> None:
        nonlocal completed
        await asyncio.sleep(0.1)
        completed = True

    async def outer() -> None:
        await scheduler.shield(inner())

    t = asyncio.create_task(outer())
    await asyncio.sleep(0)
    t.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t

    assert not completed
    assert len(scheduler._shields) == 1
    await asyncio.sleep(0.11)
    assert completed
    assert len(scheduler._shields) == 0  # type: ignore[unreachable]


async def test_wait_and_close(scheduler: Scheduler) -> None:
    inner_done = outer_done = False

    async def inner() -> None:
        nonlocal inner_done
        await asyncio.sleep(0.1)
        inner_done = True

    async def outer() -> None:
        nonlocal outer_done
        await scheduler.shield(inner())
        await asyncio.sleep(0.1)
        outer_done = True

    await scheduler.spawn(outer())
    await asyncio.sleep(0)
    assert not inner_done and not outer_done
    assert len(scheduler._shields) == 1
    assert len(scheduler._jobs) == 1

    await scheduler.wait_and_close()
    assert inner_done and outer_done  # type: ignore[unreachable]
    assert len(scheduler._shields) == 0  # type: ignore[unreachable]
    assert len(scheduler._jobs) == 0
    assert scheduler.closed


async def test_wait_and_close_timeout(scheduler: Scheduler) -> None:
    inner_done = outer_cancelled = False

    async def inner() -> None:
        nonlocal inner_done
        await asyncio.sleep(0.1)
        inner_done = True

    async def outer() -> None:
        nonlocal outer_cancelled
        await scheduler.shield(inner())
        try:
            await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            outer_cancelled = True

    await scheduler.spawn(outer())
    await asyncio.sleep(0)
    assert not inner_done and not outer_cancelled
    assert len(scheduler._shields) == 1
    assert len(scheduler._jobs) == 1

    await scheduler.wait_and_close(0.2)
    assert inner_done and outer_cancelled  # type: ignore[unreachable]
    assert len(scheduler._shields) == 0  # type: ignore[unreachable]
    assert len(scheduler._jobs) == 0
    assert scheduler.closed


async def test_wait_and_close_timeout_shield() -> None:
    inner_cancelled = outer_cancelled = False

    async def inner() -> None:
        nonlocal inner_cancelled
        try:
            await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            inner_cancelled = True
            raise

    async def outer() -> None:
        nonlocal outer_cancelled
        try:
            await scheduler.shield(inner())
        except asyncio.CancelledError:
            outer_cancelled = True

    async with Scheduler(wait_timeout=0.1) as scheduler:
        await scheduler.spawn(outer())
        await asyncio.sleep(0)
        assert not inner_cancelled and not outer_cancelled
        assert len(scheduler._shields) == 1
        assert len(scheduler._jobs) == 1

    assert inner_cancelled and outer_cancelled  # type: ignore[unreachable]
    assert len(scheduler._shields) == 0  # type: ignore[unreachable]
    assert len(scheduler._jobs) == 0
    assert scheduler.closed


async def test_wait_and_close_spawn(scheduler: Scheduler) -> None:
    another_spawned = another_done = False

    async def another() -> None:
        nonlocal another_done
        await scheduler.shield(asyncio.sleep(0.1))
        another_done = True

    async def coro() -> None:
        nonlocal another_spawned
        await asyncio.sleep(0.1)
        another_spawned = True
        await scheduler.spawn(another())

    await scheduler.spawn(coro())
    await asyncio.sleep(0)

    assert not another_spawned and not another_done
    await scheduler.wait_and_close()
    assert another_spawned and another_done  # type: ignore[unreachable]


async def test_wait_and_close_exception(make_scheduler: _MakeScheduler) -> None:
    exc_handler = mock.Mock()
    scheduler = await make_scheduler(exception_handler=exc_handler)
    exc = RuntimeError()

    async def coro() -> NoReturn:
        await asyncio.sleep(0)
        raise exc

    await scheduler.spawn(coro())
    assert len(scheduler) == 1

    await scheduler.wait_and_close()

    assert len(scheduler) == 0

    expect = {"exception": exc, "job": mock.ANY, "message": "Job processing failed"}
    if asyncio.get_running_loop().get_debug():
        expect["source_traceback"] = mock.ANY
    exc_handler.assert_called_with(scheduler, expect)


@pytest.mark.skipif(sys.version_info >= (3, 10), reason="Requires Python<3.10")
def test_scheduler_must_be_created_within_running_loop() -> None:
    with pytest.raises(RuntimeError) as exc_info:
        Scheduler(close_timeout=0, limit=0, pending_limit=0, exception_handler=None)

    assert exc_info.match("no (current|running) event loop")
