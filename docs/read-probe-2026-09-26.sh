#!/bin/bash
# Byron, September 26, 2026: "Let's probe the neuron rule as well. Space heater mode enabled."
# Continue the neuron rule's three finished 1M sweeps (mnist-1m-lowlr, -hidden, -leaky-10seed) from their checkpoints for
# 10,000 epochs each, as they ran -- paid by the evidence critic, on the code they ran on (301978a) -- and beside them a
# fresh, never-learning network of every configuration and seed, the synapse probe's two drives included: each seed's
# untrained read, for docs/synapse-matched-filter.py to set every arm against. 74 jobs, one pool of 20 workers, niceness
# 19, from the frozen worktree .claude/worktrees/probe-evidence (runs/ linked to the main checkout's), a detached
# checkout of 301978a with this script, the spec beside it and docs/synapse-read-probe.py copied in. It ran 17:53 to
# about 19:00 on September 26, 2026, all 74 jobs; the worktree was removed on September 28 once it was no longer in use.
P=/home/byron/Documents/code/walnutbutter/.claude/worktrees/probe-evidence
cd "$P"
export PYTHONPATH="$P/src"
nice -n 27 /home/byron/Documents/code/walnutbutter/.venv/bin/python docs/synapse-read-probe.py --batch docs/read-probe-2026-09-26.json --workers 20 > runs/synapse-read-probe/read-probe-2026-09-26.log 2>&1
