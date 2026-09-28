---
id: TASK-1
title: Add and remove workers from a running sweep
status: To Do
assignee: []
created_date: '2026-09-28 11:07'
labels: []
dependencies: []
references:
  - 'https://github.com/byronshock/walnutbutter/issues/30'
  - docs/rust-sweep.py
ordinal: 1000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A sweep's worker count is fixed when it starts: docs/rust-sweep.py hands its arms to a multiprocessing Pool, which can neither grow nor shrink. Changing the count today means stopping the launch script first (otherwise it moves on to its next step as though the sweep had finished), then stopping the driver and every worker by PID (an orphaned worker still running an arm would collide with its resumed copy), editing the launch script and relaunching; every running arm loses up to --checkpoint-every epochs. On September 28, 2026 that is what it took to move the read-window sweep (mnist-100k-hidden60-window-whitened) from four workers to five, and Byron asked for sweeps whose workers can be added and removed while they run. The probe driver (docs/synapse-read-probe.py --batch) has the same fixed pool. First filed as GitHub issue #30, before Backlog.md was set up here.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Raising the worker count of a running sweep starts the extra workers on arms not yet begun, without stopping or restarting any running arm
- [ ] #2 Lowering the worker count retires workers without losing finished work: a retired worker takes no new arm, and an arm it leaves unfinished resumes exactly from its checkpoint (AUTHORITY.md 12.11)
- [ ] #3 No arm ever runs in two processes at once, and every arm's results are bit-identical to the same arm run at a fixed worker count
- [ ] #4 The sweep's log records each change of worker count and when it took effect
- [ ] #5 How to change the count of a running sweep is documented in docs/rust-sweep.py
- [ ] #6 Tests cover raising the count, lowering it, and an arm resumed after its worker was retired
<!-- AC:END -->
