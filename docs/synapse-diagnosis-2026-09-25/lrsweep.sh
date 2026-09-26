#!/bin/bash
# plan §5.3, Byron's go of September 25: the learning rate re-found under exploration at the synapse, mnist's point
S=/tmp/claude-1000/-home-byron-Documents-code-walnutbutter/5e969395-0125-4ccd-9daf-efdcffa9eda4/scratchpad
M=/home/byron/Documents/code/walnutbutter/.claude/worktrees/synapse-measure
cd $M
export PYTHONPATH=$S/site-measure-94dd249:$M/src
PY="nice -n 27 /home/byron/Documents/code/walnutbutter/.venv/bin/python docs/rust-sweep.py"
POINT="--problem mnist --interval 100 --threshold 0.6 --minimum-potential -0.2 --hidden-neurons 0 --temperature 2 --eligibility hazard --seed 1 2 3 --epochs 25000 --checkpoint-every 5000"
$PY --name synapse-lr-25k $POINT --exploration synapse --synapse-hazard 0.01 --hazard-family loglinear --synapse-scaling count --trace-counts all --drive rate charged --lr 0.0005 0.001 0.002 0.005 0.01 --workers 24 > $S/synapse-lr-25k.log 2>&1 &
$PY --name neuron-lr-25k $POINT --exploration neuron --delta 0.3125 --drive rate --lr 0.0005 0.002 --workers 6 > $S/neuron-lr-25k.log 2>&1 &
wait
echo done
