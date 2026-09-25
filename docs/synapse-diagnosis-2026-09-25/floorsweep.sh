#!/bin/bash
# Byron, Sept 25: the synapse rule at 200k epochs with the §8.16 trap out of reach (floor -1.2) and at the pinned floor,
# lr step-matched to the neuron rule's 0.0005/0.001; the neuron rule at the deep floor as the reference
S=/tmp/claude-1000/-home-byron-Documents-code-walnutbutter/5e969395-0125-4ccd-9daf-efdcffa9eda4/scratchpad
M=/home/byron/Documents/code/walnutbutter/.claude/worktrees/synapse-measure
cd $M
export PYTHONPATH=$S/site-measure-94dd249:$M/src
PY="nice -n 27 /home/byron/Documents/code/walnutbutter/.venv/bin/python docs/rust-sweep.py"
POINT="--problem mnist --interval 100 --threshold 0.6 --hidden-neurons 0 --temperature 2 --eligibility hazard --seed 1 2 3 --epochs 200000 --checkpoint-every 5000 --trace-every 5000"
$PY --name synapse-floor-200k $POINT --minimum-potential -0.2 -1.2 --exploration synapse --synapse-hazard 0.01 --hazard-family loglinear --synapse-scaling count --trace-counts all --drive rate --lr 0.002 0.005 --workers 12 > $S/synapse-floor-200k.log 2>&1 &
$PY --name neuron-floor-200k $POINT --minimum-potential -1.2 --exploration neuron --delta 0.3125 --drive rate --lr 0.0005 --workers 3 > $S/neuron-floor-200k.log 2>&1 &
wait
echo done
