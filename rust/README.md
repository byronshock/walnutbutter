# walnutbutter-schedule

The wave loop of `propagation.Schedule.run` (AUTHORITY.md §4.4) in Rust, behind PyO3.

## Why

Profiling the array engine showed the arithmetic is **1.3% of the runtime**: the sparse
matrix-vector product that performs every signal delivery costs 42 ms out of 3,271. The
rest is Python interpreter overhead, because the network is small (60 neurons, 766
connections) and numpy's per-call overhead dwarfs 60 doubles of work. A compiled loop
that owns the state across the run avoids both the interpreter and the per-call overhead.

## Build

Needs a Rust toolchain, which this machine does not have:

    sudo apt install rustc cargo        # Pop!_OS 24.04 ships 1.95
    .venv/bin/pip install maturin
    cd rust && ../.venv/bin/maturin develop --release

Then `tests/test_fast.py` runs; it is skipped when the module is absent.

## What it implements

The epoch's wave loop exactly as §4.4 specifies it: events batched into waves by the
clock's slack, an input's exact time anchoring the wave it joins, every signal of a wave
delivered and settled before any neuron of that wave decides to fire, forced neurons
before touched ones, then every other neuron checked for a threshold that has fallen with
its silence (§5.4). Plus the two local rules that must run inside the loop because they
change weights the rest of the epoch sees: the quash (§6.11) and leaky Hebb (§6.12), and
the eligibility a teacher pays at the read (§6.9) -- the same tally the epoch form of the
centred Hebbian eligibility of §6.7 reads (`reinforce_count_hebb`, `reinforce_hebb` from
September 16 to 17, 2026; the ±1 rule it replaced is `reinforce_wrong_hebb`). The
single-spike rule of §6.7, hebb and hazard alike, is charged in the loop at every decision,
settled with `settle_scores` and paid by `reinforce_scores`.

Exploration noise (§6.1) and the perturb eligibility of §6.7 as well, since September
14, 2026: the engine runs Python's own MT19937, seeded by handing over `rng.getstate()`,
so it draws the same uniforms in the same order as the other two engines and pairs them
by the same Box-Muller, and `fast.sync_explore` hands the state back so Python's stream
carries on from where Rust left it. No per-wave round trip, and the draws are equal, not
approximately equal: `tests/test_fast.py` checks them with `==`.

The Teacher's firing-rate memory, homeostasis and un-sticking (§1.3) are mirrored by
`fast.train` once an epoch in Python, in the object engine's order, at the constants the
command line runs them at -- so a Rust run is the run `walnutbutter --seeds` would do, and
`tests/test_fast.py` checks the moved thresholds against the object engine every epoch.
`docs/rust-sweep.py` builds goo as well as the grid. This is the default engine for a sweep.

**A stamp that never repeats.** The neurons a wave touched are marked so each is checked
once per wave; the mark was the epoch's wave number, which `reset()` zeroed, so a neuron last
touched in wave *k* of an earlier epoch passed for touched in wave *k* of a later one and was
never asked whether it fired. Found September 14, 2026 on goo with no direct projection, where
an output can go a whole epoch untouched; the mark is now a counter that never resets, as
`propagation.py`'s is. Runs before that fix could miss a firing, reproducibly.

## What it does not

- **The dopamine rule's in-loop weight updates** (§6.2–6.6, mode "apply"). The teacher,
  adaline, reinforce and local rules all pay at the read, in Python, which is once an
  epoch rather than once a wave.
