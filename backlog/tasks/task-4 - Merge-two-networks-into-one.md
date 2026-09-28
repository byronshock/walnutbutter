---
id: TASK-4
title: Merge two networks into one
status: To Do
assignee: []
created_date: '2026-09-28 11:24'
labels: []
dependencies: []
references:
  - AUTHORITY.md
ordinal: 4000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Byron, September 28, 2026: "In addition to the resizing task, I have a related task that is *merging* two networks." It is related to TASK-3, which adds and removes neurons one at a time in a living network: a merge brings a whole second network in at once, its neurons, synapses and learned state. So far only the aim is stated; what a merge means is Byron's to specify, and it has to be settled before code: which zones the merged network has (the two input and output zones side by side, one network feeding the other, or otherwise); whether synapses are drawn between the two networks, by what rule and from what seed; how their state combines (clocks, random streams, signals in flight, reinforcement baselines, the matched filters' books); and whether living networks merge while they run or checkpoints merge when loaded.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 What a merge does is specified in AUTHORITY.md before it is built: the merged zones, any synapses drawn between the two networks, and how their state combines
- [ ] #2 Two networks can be merged into one that runs, each keeping its neurons, synapses, weights and learned state
- [ ] #3 A merge is recorded (which networks, and any draw it took) and reproducible: a resumed merged network is the same run continued (AUTHORITY.md 12.11)
- [ ] #4 The three engines agree on a merged network (fast.compare)
- [ ] #5 Tests cover merging two networks and running the result
<!-- AC:END -->
