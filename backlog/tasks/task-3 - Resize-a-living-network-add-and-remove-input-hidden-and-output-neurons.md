---
id: TASK-3
title: 'Resize a living network: add and remove input, hidden and output neurons'
status: To Do
assignee: []
created_date: '2026-09-28 11:23'
labels: []
dependencies:
  - TASK-2
references:
  - 'https://github.com/byronshock/walnutbutter/issues/14'
  - AUTHORITY.md
ordinal: 3000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Byron, September 28, 2026: he wants to dynamically resize a living network, adding and removing input, hidden and output neurons. It builds on the living goo of TASK-2, and widens GitHub issue #14 (September 16: grow a checkpointed goo by N hidden neurons when it is loaded, --request_more_goo N) to all three zones, to removal as well as addition, and to a network that is running. It must keep two design values: neurons are first-class citizens (AUTHORITY.md 0.2: a zone decides where the drive reaches and where the read counts, and nothing about a neuron's own rules), and structures are permissive, never artificially restricted. What stands in the way is in the specification: zones are addressed by index (4.3: the first neurons are the input zone and the last ones the output zone), mnist fixes its zones (11.5: 395 inputs; 11.6: 60 outputs, complement-coded), and the matched filter's book is sized to the output zone (9.4). Resizing the input and output zones therefore needs clauses before code.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Hidden neurons can be added to and removed from a living network while it runs, without stopping it
- [ ] #2 Input and output neurons can be added and removed while it runs; the drive and the read follow the zones as they stand, and the critic reads a resized output zone
- [ ] #3 A new neuron is wired by the rule the network was built under, and every resize (what, when, and any draw it took) is recorded so that a resumed run reproduces it exactly (AUTHORITY.md 12.11)
- [ ] #4 A removed neuron leaves nothing behind: its synapses, its signals in flight, and every record that refers to it
- [ ] #5 The three engines agree on a resized network (fast.compare)
- [ ] #6 Checkpoints carry the resize history, and tests cover adding and removing neurons in each of the three zones
<!-- AC:END -->
