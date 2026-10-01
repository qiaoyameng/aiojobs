import asyncio
import gc
from collections.abc import Awaitable
from typing import Any, Callable, Dict, List, NoReturn
from unittest import mock

import pytest

from aiojobs import Group, GroupError, Job, Scheduler

_MakeScheduler = Callable[..., Awaitable[Scheduler]]


async def test_create_group(scheduler: Scheduler) -> None:
    group = scheduler.create_group()
    assert isinstance(group, Group)
    assert group.scheduler is scheduler
    assert not group.closed
    assert len(group) == 0


async def test_group_spawn_and_membership(scheduler: Scheduler) -> None:
    async def coro() -> None:
        await asyncio.sleep(1)

    group = scheduler.create_group()
    job = await group.spawn(coro())
    assert not job.closed
    assert len(group) == 1
    assert list(group) == [job]
    assert job in group
    assert job in scheduler
    assert "jobs=1" in repr(group)
    await group.close()
    assert group.closed
    assert "closed" in repr(group)


async def test_group_normal_exit_waits_for_all_members(
    make_scheduler: _MakeScheduler,
) -> None:
    scheduler = await make_scheduler(close_timeout=1.0, limit=2)
    done: List[int] = []

    async def worker(value: int) -> int:
        await asyncio.sleep(0.01)
        done.append(value)
        return value

    async with scheduler.create_group() as group:
        jobs = [await group.spawn(worker(i)) for i in range(4)]

    assert group.closed
    assert sorted(done) == [0, 1, 2, 3]
    assert [await job.wait() for job in jobs] == [0, 1, 2, 3]


async def test_group_failure_cancels_members_and_keeps_other_groups(
    make_scheduler: _MakeScheduler,
) -> None:
    exc_handler = mock.Mock()
    scheduler = await make_scheduler(
        close_timeout=1.0, limit=2, exception_handler=exc_handler
    )
    loop = asyncio.get_running_loop()
    loop_contexts: List[Dict[str, Any]] = []
    loop.set_exception_handler(lambda loop, ctx: loop_contexts.append(ctx))

    never_started = False

    async def failer() -> NoReturn:
        await asyncio.sleep(0.01)
        raise RuntimeError("first failure")

    async def cleanup_failer() -> None:
        try:
            await asyncio.sleep(10)
        finally:
            raise RuntimeError("cleanup failure")

    async def queued() -> None:
        nonlocal never_started
        never_started = True

    async def worker(value: int) -> int:
        await asyncio.sleep(0.01)
        return value

    group_a = scheduler.create_group()
    group_b = scheduler.create_group()

    job_fail = await group_a.spawn(failer(), name="failer")
    job_cleanup = await group_a.spawn(cleanup_failer(), name="cleanup")
    job_queued = await group_a.spawn(queued(), name="queued")
    jobs_b = [await group_b.spawn(worker(i)) for i in range(3)]

    with pytest.raises(GroupError) as exc_info:
        await group_a.wait()

    # every actually failed job is reported, the first failure included
    err = exc_info.value
    assert set(err.failures) == {job_fail, job_cleanup}
    assert str(err.failures[job_fail]) == "first failure"
    assert str(err.failures[job_cleanup]) == "cleanup failure"

    # the pending member was discarded without being executed
    assert job_queued.closed
    assert not never_started

    # the other group is unaffected and its results are complete
    assert [await job.wait() for job in jobs_b] == [0, 1, 2]
    await group_b.wait()

    await group_a.close()
    await group_b.close()

    # no duplicate failure reports and no unretrieved exceptions
    exc_handler.assert_not_called()
    gc.collect()
    await asyncio.sleep(0)
    assert loop_contexts == []


async def test_group_aexit_body_exception_is_not_masked(
    scheduler: Scheduler,
) -> None:
    cancelled = False

    async def cleanup_failer() -> None:
        nonlocal cancelled
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled = True
            raise RuntimeError("cleanup failure")

    with pytest.raises(ValueError, match="body"):
        async with scheduler.create_group() as group:
            await group.spawn(cleanup_failer())
            await asyncio.sleep(0)
            raise ValueError("body")

    assert cancelled
    assert group.closed


async def test_group_spawn_after_close(scheduler: Scheduler) -> None:
    async def coro() -> None:
        pass

    group = scheduler.create_group()
    await group.close()

    coro_instance = coro()
    with pytest.raises(RuntimeError):
        await group.spawn(coro_instance)

    with pytest.warns(RuntimeWarning):
        del coro_instance


async def test_group_close_is_idempotent(scheduler: Scheduler) -> None:
    group = scheduler.create_group()
    await group.close()
    await group.close()
    assert group.closed


async def test_group_wait_is_noop_without_failures(scheduler: Scheduler) -> None:
    async def coro() -> int:
        await asyncio.sleep(0)
        return 42

    group = scheduler.create_group()
    job = await group.spawn(coro())
    await group.wait()
    assert await job.wait() == 42
    await group.close()


async def test_group_jobs_are_tracked_by_scheduler(
    make_scheduler: _MakeScheduler,
) -> None:
    scheduler = await make_scheduler(close_timeout=1.0, limit=1)

    async def coro() -> None:
        await asyncio.sleep(1)

    group = scheduler.create_group()
    job1 = await group.spawn(coro())
    job2 = await group.spawn(coro())
    assert job1.active
    assert job2.pending
    assert scheduler.active_count == 1
    assert scheduler.pending_count == 1
    await group.close()
