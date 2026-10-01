import asyncio
from typing import Any, List, NoReturn
from unittest import mock

import pytest

from aiojobs import JobGroupError, Scheduler


async def test_group_waits_for_all_jobs_and_returns_results() -> None:
    async with Scheduler(limit=1) as scheduler:
        group = scheduler.create_group()

        async def job(value: int) -> int:
            await asyncio.sleep(0)
            return value

        await group.spawn(job(1), name="first")
        await group.spawn(job(2), name="second")

        assert await group.wait() == [1, 2]
        assert group.closed


async def test_group_failure_cancels_group_but_not_other_groups() -> None:
    handler = mock.Mock()
    scheduler = Scheduler(limit=3, close_timeout=1, exception_handler=handler)
    failed_group = scheduler.create_group()
    healthy_group = scheduler.create_group()

    running_started = asyncio.Event()
    queued_started = asyncio.Event()
    cleanup_started = asyncio.Event()
    failing_ready = asyncio.Event()
    release_failure = asyncio.Event()
    initial_error = RuntimeError("initial failure")
    cleanup_error = RuntimeError("cleanup failure")

    async def running() -> None:
        running_started.set()
        await asyncio.Event().wait()

    async def failing() -> NoReturn:
        failing_ready.set()
        await release_failure.wait()
        raise initial_error

    async def cleanup_error_job() -> NoReturn:
        cleanup_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            raise cleanup_error

    async def queued() -> None:
        queued_started.set()

    async def healthy(value: str) -> str:
        await asyncio.sleep(0)
        return value

    running_job = await failed_group.spawn(running(), name="running")
    await failed_group.spawn(failing(), name="failing")
    await failed_group.spawn(cleanup_error_job(), name="cleanup")
    await running_started.wait()
    await cleanup_started.wait()
    await failed_group.spawn(queued(), name="queued")
    await failing_ready.wait()
    release_failure.set()

    healthy_one = await healthy_group.spawn(healthy("one"), name="healthy-one")
    healthy_two = await healthy_group.spawn(healthy("two"), name="healthy-two")

    with pytest.raises(JobGroupError) as exc_info:
        await failed_group.wait()

    errors = exc_info.value.errors
    assert [job.get_name() for job, _ in errors] == ["failing", "cleanup"]
    assert [error for _, error in errors] == [initial_error, cleanup_error]
    assert running_job.closed
    assert not queued_started.is_set()
    assert await healthy_group.wait() == ["one", "two"]
    assert await healthy_one.wait() == "one"
    assert await healthy_two.wait() == "two"
    assert scheduler.active_count == 0
    assert scheduler.pending_count == 0
    assert not handler.called

    await scheduler.close()


async def test_group_rejects_spawn_after_wait() -> None:
    scheduler = Scheduler()
    group = scheduler.create_group()

    async def noop() -> None:
        pass

    await group.spawn(noop())
    await group.wait()

    with pytest.raises(RuntimeError, match="after closing group"):
        await group.spawn(noop())

    await scheduler.close()


async def test_group_close_rejects_future_spawn_and_cancels_jobs() -> None:
    scheduler = Scheduler()
    group = scheduler.create_group()
    started = asyncio.Event()

    async def job() -> None:
        started.set()
        await asyncio.Event().wait()

    await group.spawn(job())
    await started.wait()
    await group.close()

    assert group.closed

    async def noop() -> None:
        pass

    with pytest.raises(RuntimeError, match="after closing group"):
        await group.spawn(noop())

    await scheduler.close()


async def test_scheduler_close_closes_open_groups() -> None:
    scheduler = Scheduler()
    group = scheduler.create_group()
    started = asyncio.Event()

    async def job() -> None:
        started.set()
        await asyncio.Event().wait()

    await group.spawn(job())
    await started.wait()
    await scheduler.close()

    assert group.closed
    assert scheduler.active_count == 0


async def test_group_failure_cancels_spawn_waiting_for_pending_slot() -> None:
    scheduler = Scheduler(limit=2, pending_limit=1)
    group = scheduler.create_group()
    blocker_started = asyncio.Event()
    blocked_started = asyncio.Event()
    failing_ready = asyncio.Event()
    release_failure = asyncio.Event()
    initial_error = RuntimeError("initial failure")

    async def blocker() -> None:
        blocker_started.set()
        await asyncio.Event().wait()

    async def queued() -> None:
        await asyncio.Event().wait()

    async def blocked() -> None:
        blocked_started.set()

    async def failing() -> NoReturn:
        failing_ready.set()
        await release_failure.wait()
        raise initial_error

    await group.spawn(blocker(), name="blocker")
    await group.spawn(failing(), name="failing")
    await group.spawn(queued(), name="queued")
    blocked_spawn = asyncio.ensure_future(
        group.spawn(blocked(), name="blocked")
    )
    await asyncio.sleep(0)
    assert not blocked_spawn.done()
    await failing_ready.wait()
    release_failure.set()

    with pytest.raises(asyncio.CancelledError):
        await blocked_spawn

    with pytest.raises(JobGroupError) as exc_info:
        await group.wait()
    assert [job.get_name() for job, _ in exc_info.value.errors] == ["failing"]
    assert not blocked_started.is_set()
    assert scheduler.active_count == 0
    assert scheduler.pending_count == 0
    await scheduler.close()


async def test_group_context_manager_returns_results() -> None:
    async with Scheduler() as scheduler:
        group = scheduler.create_group()
        async with group:
            await group.spawn(asyncio.sleep(0, result="value"))
        assert await group.wait() == ["value"]
