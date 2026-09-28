---
id: TASK-2
title: Create a living goo that runs until stopped
status: To Do
assignee: []
created_date: '2026-09-28 11:22'
labels: []
dependencies: []
references:
  - AUTHORITY.md
  - docs/rust-sweep.py
  - src/walnutbutter/cli.py
ordinal: 2000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Byron, September 28, 2026: he wants an option to create living goo that runs until stopped. AUTHORITY.md states the principle (0.8 and 12.12: the network keeps living; there is no training run and no evaluation run, only one run that keeps going, and no rule may assume an end), but every run today is bounded: docs/rust-sweep.py takes --epochs (1,000,000 by default) and the walnutbutter command runs --epochs under --headless (1 by default), on the objects and arrays engines only, not the Rust loop that long runs use. Results of a living run are read as health, not convergence: drift is normal, and figures accumulated "to date" say less and less as the run goes on.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 An option creates a goo that runs epoch after epoch with no epoch limit until it is stopped
- [ ] #2 Stopping it (Ctrl+C or a signal) ends it at an epoch boundary with a checkpoint written, and resuming from that checkpoint is the same run continued (AUTHORITY.md 12.11)
- [ ] #3 While it runs it writes checkpoints at a set cadence, so a crash or reboot costs at most that many epochs
- [ ] #4 While it runs it reports its health at a set cadence over recent epochs (the fraction right, the reinforcement, stuck counts), not only totals to date
- [ ] #5 It runs on the Rust engine
- [ ] #6 The option is documented, and tests cover stopping, resuming, and a resumed living run matching the uninterrupted one
<!-- AC:END -->
