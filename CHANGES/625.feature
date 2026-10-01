Added ``Scheduler.create_group()`` for spawning jobs as a group: the
first failing member cancels the running ones and discards the pending
ones, while ``Group.wait()`` raises ``GroupError`` reporting every
failed job.
