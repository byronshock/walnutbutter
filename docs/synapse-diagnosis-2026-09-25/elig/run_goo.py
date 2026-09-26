"""Goo 60 on the copy problem, exploring at the synapse (§7.5, §8.16), a Teacher paying the reinforce rule, run under
fast.compare so the object engine and Rust are held to each other with == after every epoch, with the object engine's
_decide_synapses instrumented (instrument.py). usage: run_goo.py SEED EPOCHS WEIGHT(none|float) LR"""
import json, math, random, sys, time
import numpy as np
sys.path.insert(0, "/tmp/claude-1000/-home-byron-Documents-code-walnutbutter/5e969395-0125-4ccd-9daf-efdcffa9eda4/scratchpad/diag/elig")
from instrument import Instrument, XBINS, MBINS
from walnutbutter import fast
from walnutbutter.goo import Goo
from walnutbutter.learning import Teacher
from walnutbutter.neuron import Neuron

OUT = "/tmp/claude-1000/-home-byron-Documents-code-walnutbutter/5e969395-0125-4ccd-9daf-efdcffa9eda4/scratchpad/diag/elig"
seed, epochs = int(sys.argv[1]), int(sys.argv[2])
weight = None if sys.argv[3] == "none" else float(sys.argv[3])
lr = float(sys.argv[4])
Neuron.verbose = False
g = Goo(count=60, across=8, seed=seed, weight=weight)
g.rule, g.drive, g.read = "reinforce", "rate", "count"
g.set_exploration("synapse")
ins = Instrument(g)
teacher = Teacher(g, seed=seed + 100, rule="reinforce", eligibility="hazard", target="copy", lr=lr)
t0 = time.time()
parted = fast.compare(g, epochs=epochs, teacher=teacher)
ins.finish()
W = ins.wave_array()
rows = np.array(ins.rows)
tag = f"goo60-seed{seed}-w{sys.argv[3]}-lr{lr:g}-{epochs}"
np.savez_compressed(f"{OUT}/{tag}.npz", waves=W, rows=rows, xb=ins.xb, mb=ins.mb, cal=ins.cal, cal_u=ins.cal_u)
print(json.dumps({"tag": tag, "parted": parted[:3], "bad": ins.bad[:5], "n_bad": len(ins.bad), "epochs": len(rows),
                  "waves": len(W), "seconds": round(time.time() - t0),
                  "synapses": len(ins.conns), "outputs": len(g.output_row())}))
