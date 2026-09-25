#!/bin/bash
# plan §5.2: smoke and resume rehearsal at mnist's operating point, exploration at the synapse, from the frozen worktree
S=/tmp/claude-1000/-home-byron-Documents-code-walnutbutter/5e969395-0125-4ccd-9daf-efdcffa9eda4/scratchpad
M=/home/byron/Documents/code/walnutbutter/.claude/worktrees/synapse-measure
cd $M
export PYTHONPATH=$S/site-measure-94dd249:$M/src
PY="nice -n 27 /home/byron/Documents/code/walnutbutter/.venv/bin/python docs/rust-sweep.py"
KNOBS="--problem mnist --interval 100 --threshold 0.6 --minimum-potential -0.2 --hidden-neurons 0 --temperature 2 --lr 0.002 --eligibility hazard --exploration synapse --drive rate charged --seed 1 2 3"
$PY --name smoke-synapse-3k $KNOBS --epochs 3000 --checkpoint-every 500 --workers 6 > $S/smoke-3k.log 2>&1 &
A=$!
$PY --name smoke-synapse-cut $KNOBS --epochs 1500 --workers 6 > $S/smoke-cut.log 2>&1
$PY --name smoke-synapse-cut2 $KNOBS --epochs 1500 --resume-from smoke-synapse-cut --workers 6 > $S/smoke-cut2.log 2>&1
wait $A
echo done
