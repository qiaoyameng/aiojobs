import asyncio
import sys
from collections.abc import Awaitable, Callable
from typing import List

import pytest

from aiojobs import Scheduler

if sys.version_info >= (3, 11):
    from asyncio import timeout as asyncio_timeout
else:
    from async_timeout import timeout as asyncio_timeout

_MakeScheduler = Callable[..., Awaitable[Scheduler]]


class FakeTimerHandle:
    def __init__(self, when: float, callback: Callable[[], None]) -> None:
        self._when = when
        self._callback = callback
        self.cancelled = False
        self.fired = False

    def when(self) -> float:
        return self._when

    def cancel(self) -> None:
        self.cancelled = True

    def fire(self) -> None:
        self.fired = True
        self._callback()


class FakeClock:
    """A controllable replacement for the loop monotonic clock."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.handles: List[FakeTimerHandle] = []

    def monotonic(self) -> float:
        return self.now

    def call_at(self, when: float, callback: Callable[[], None]) -> FakeTimerHandle:
        handle = FakeTimerHandle(when, callback)
        self.handles.append(handle)
        return handle

    def advance(self, delay: float) -> None:
        self.now += delay
        self.fire_due()

    def fire_due(self) -> None:
        while True:
            due = [
                handle
                for handle in self.handles
                if not handle.cancelled
                and not handle.fired
                and handle.when() <= self.now
            ]
            if not due:
                return
            for handle in due:
                handle.fire()

    @property
    def last_handle(self) -> FakeTimerHandle:
        return self.handles[-1]


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    clock = FakeClock()
    monkeypatch.setattr(Scheduler, "_monotonic", lambda self: clock.monotonic())
    monkeypatch.setattr(
        Scheduler,
        "_call_at",
        lambda self, when, callback: clock.call_at(when, callback),
    )
    return clock


async def tick() -> None:
    # let spawned tasks and their done callbacks run
    for _ in range(5):
        await asyncio.sleep(0)


async def test_spawn_at_not_active_before_deadline(
    make_scheduler: _MakeScheduler, clock: FakeClock
) -> None:
    scheduler = await make_scheduler()
    started = False

    async def coro() -> None:
        nonlocal started
        started = True

    job = await scheduler.spawn_at(clock.now + 10, coro())

    assert len(scheduler) == 1
    assert scheduler.active_count == 0
    assert scheduler.pending_count == 1
    assert job.pending
    assert not job.active

    clock.advance(9)
    await tick()
    assert not started
    assert scheduler.active_count == 0
    assert scheduler.pending_count == 1

    clock.advance(1)
    await tick()
    assert started
    assert job.closed
    assert scheduler.active_count == 0
    assert scheduler.pending_count == 0
    assert len(scheduler) == 0


async def test_spawn_at_earlier_deadline_reschedules_wakeup(
    make_scheduler: _MakeScheduler, clock: FakeClock
) -> None:
    scheduler = await make_scheduler()
    t0 = clock.now
    started: List[str] = []

    async def coro(tag: str) -> None:
        started.append(tag)

    late = await scheduler.spawn_at(t0 + 100, coro("late"))
    first_handle = clock.last_handle
    assert first_handle.when() == t0 + 100

    early = await scheduler.spawn_at(t0 + 10, coro("early"))
    # inserting a closer deadline moves the wakeup earlier
    assert first_handle.cancelled
    assert clock.last_handle.when() == t0 + 10

    clock.advance(10)
    await tick()
    assert started == ["early"]
    assert early.closed
    assert late.pending

    clock.advance(90)
    await tick()
    assert started == ["early", "late"]


async def test_spawn_at_same_deadline_submission_order(
    make_scheduler: _MakeScheduler, clock: FakeClock
) -> None:
    scheduler = await make_scheduler(limit=1)
    started: List[int] = []

    async def coro(num: int) -> None:
        started.append(num)

    when = clock.now + 5
    jobs = [await scheduler.spawn_at(when, coro(num)) for num in range(3)]

    clock.advance(5)
    # limit=1 starts the jobs one by one, in submission order
    async with asyncio_timeout(1):
        await asyncio.gather(*(job.wait() for job in jobs))
    assert started == [0, 1, 2]
    assert len(scheduler) == 0


async def test_spawn_at_respects_limit_when_due(
    make_scheduler: _MakeScheduler, clock: FakeClock
) -> None:
    scheduler = await make_scheduler(limit=1)
    fut: asyncio.Future[None] = asyncio.Future()
    started = False

    async def blocker() -> None:
        await fut

    async def coro() -> None:
        nonlocal started
        started = True

    holding = await scheduler.spawn(blocker())
    job = await scheduler.spawn_at(clock.now + 5, coro())

    clock.advance(5)
    await tick()
    # the job is due but the concurrency slot is still occupied
    assert not started
    assert job.pending
    assert scheduler.active_count == 1
    assert scheduler.pending_count == 1

    fut.set_result(None)
    await holding.wait()
    await tick()
    assert started
    assert job.closed


async def test_spawn_at_cancelled_job_not_resurrected(
    make_scheduler: _MakeScheduler, clock: FakeClock
) -> None:
    scheduler = await make_scheduler()
    ran = False

    async def coro() -> None:
        nonlocal ran
        ran = True

    job = await scheduler.spawn_at(clock.now + 3600, coro())
    await job.close()
    await tick()
    assert job.closed
    assert scheduler.pending_count == 0
    assert len(scheduler) == 0

    # the stale wakeup fires much later and must not revive the job
    clock.advance(3600)
    await tick()
    assert not ran
    assert len(scheduler) == 0


async def test_close_does_not_wait_for_far_deadline(
    make_scheduler: _MakeScheduler, clock: FakeClock
) -> None:
    scheduler = await make_scheduler()
    ran = False

    async def coro() -> None:
        nonlocal ran
        ran = True

    job = await scheduler.spawn_at(clock.now + 3600, coro())
    handle = clock.last_handle

    async with asyncio_timeout(1):
        await scheduler.close()

    assert scheduler.closed
    assert job.closed
    assert handle.cancelled
    assert scheduler.pending_count == 0
    assert len(scheduler) == 0

    clock.advance(3600)
    await tick()
    assert not ran


async def test_close_cancels_last_scheduled_wakeup(
    make_scheduler: _MakeScheduler, clock: FakeClock
) -> None:
    scheduler = await make_scheduler()
    t0 = clock.now
    ran: List[str] = []

    async def coro(tag: str) -> None:
        ran.append(tag)

    await scheduler.spawn_at(t0 + 5, coro("soon"))
    await scheduler.spawn_at(t0 + 3600, coro("later"))

    clock.advance(5)
    await tick()
    assert ran == ["soon"]
    # the wakeup was moved to the next pending deadline
    last_handle = clock.last_handle
    assert last_handle.when() == t0 + 3600
    assert not last_handle.cancelled

    async with asyncio_timeout(1):
        await scheduler.close()
    assert last_handle.cancelled

    clock.advance(3600)
    await tick()
    assert ran == ["soon"]


async def test_wait_and_close_covers_delayed_jobs(
    make_scheduler: _MakeScheduler, clock: FakeClock
) -> None:
    scheduler = await make_scheduler()
    finished = False

    async def coro() -> None:
        nonlocal finished
        finished = True

    await scheduler.spawn_at(clock.now + 5, coro())
    closing = asyncio.ensure_future(scheduler.wait_and_close())
    await tick()
    assert not closing.done()
    assert not scheduler.closed

    clock.advance(5)
    async with asyncio_timeout(1):
        await closing
    assert finished
    assert scheduler.closed


async def test_job_wait_covers_delayed_job(
    make_scheduler: _MakeScheduler, clock: FakeClock
) -> None:
    scheduler = await make_scheduler()

    async def coro() -> int:
        return 42

    job = await scheduler.spawn_at(clock.now + 5, coro())
    waiter = asyncio.ensure_future(job.wait())
    await tick()
    assert not waiter.done()

    clock.advance(5)
    async with asyncio_timeout(1):
        result = await waiter
    assert result == 42


async def test_spawn_at_counts_towards_pending_limit(
    make_scheduler: _MakeScheduler, clock: FakeClock
) -> None:
    scheduler = await make_scheduler(limit=1, pending_limit=1)
    fut: asyncio.Future[None] = asyncio.Future()

    async def blocker() -> None:
        await fut

    async def coro() -> None:
        pass

    await scheduler.spawn(blocker())
    job = await scheduler.spawn_at(clock.now + 100, coro())
    assert scheduler.active_count == 1
    assert scheduler.pending_count == 1

    spawned = False

    async def try_spawn() -> None:
        nonlocal spawned
        await scheduler.spawn_at(clock.now + 200, coro())
        spawned = True

    task = asyncio.ensure_future(try_spawn())
    await tick()
    # no free pending capacity while the delayed job holds its slot
    assert not spawned

    await job.close()
    await tick()
    async with asyncio_timeout(1):
        await task
    assert spawned
    assert scheduler.pending_count == 1

    fut.set_result(None)


async def test_spawn_at_past_deadline_starts_immediately(
    make_scheduler: _MakeScheduler, clock: FakeClock
) -> None:
    scheduler = await make_scheduler()
    started = False

    async def coro() -> None:
        nonlocal started
        started = True

    job = await scheduler.spawn_at(clock.now - 1, coro())
    assert job.active
    assert scheduler.active_count == 1
    assert scheduler.pending_count == 0
    await tick()
    assert started


async def test_spawn_at_after_close(
    make_scheduler: _MakeScheduler, clock: FakeClock
) -> None:
    scheduler = await make_scheduler()

    async def coro() -> None:
        pass

    await scheduler.close()

    coro_instance = coro()
    with pytest.raises(RuntimeError):
        await scheduler.spawn_at(clock.now + 5, coro_instance)

    with pytest.warns(RuntimeWarning):
        del coro_instance
