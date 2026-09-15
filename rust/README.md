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
the eligibility a teacher pays at the read (§6.9).

Exploration noise (§6.1) and the perturb eligibility of §6.7 as well, since September
14, 2026: the engine runs Python's own MT19937, seeded by handing over `rng.getstate()`,
so it draws the same uniforms in the same order as the other two engines and pairs them
by the same Box-Muller, and `fast.sync_explore` hands the state back so Python's stream
carries on from where Rust left it. No per-wave round trip, and the draws are equal, not
approximately equal: `tests/test_fast.py` checks them with `==`.

## What it does not

- **The dopamine rule's in-loop weight updates** (§6.2–6.6, mode "apply"). The teacher,
  adaline, reinforce and local rules all pay at the read, in Python, which is once an
  epoch rather than once a wave.
