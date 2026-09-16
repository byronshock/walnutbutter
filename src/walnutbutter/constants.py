"""Every global constant of walnutbutter, in one place.

These are the defaults the constructors, the neuron's clock, the Teacher and
the command line all read from: change a value here and every path follows.
They are the physics of the substance (what a neuron needs to fire, how fast
it leaks, how learning moves a weight) and the shape of the default network.
Lattice geometry (cell spacing, row spacing, the guaranteed radius of a
column) is not tunable and stays with the lattice that owns it; the format
fallbacks for old checkpoints stay in persistence.py, because they record
what those files meant when they were written, not what the default is now.

The clock values live here but run from `Neuron.refractory` and
`Neuron.refractory_hops`, the class attributes the command line sets (and
restores) per run.
"""

# --- the default network: footprint and wiring ----------------------------------
ACROSS = 8  # cells across (the input row has one neuron per coded bit)
ROWS = 10  # rows of cells, input at the bottom, output at the top
OMEGA = 0.2  # proportion of all connections that are small-world shortcuts, 0 <= omega < 1
REACH = 2.0  # lattice wiring: every pair within this many unit distances connects (the two hex rings)
WEIGHT_RANGE = (-1.0, 1.0)  # random weights are drawn from this range, and learning clips to it
WEIGHT_EPSILON = 0.001  # --epsilon: the smallest weight allowed under --positive-weights, range (epsilon, 1)

# --- the neuron: activation and the clock (nominal milliseconds) ----------------
THRESHOLD = 0.25  # total weighted input a neuron needs before it fires, quoted at THRESHOLD_FAN_IN incoming synapses
THRESHOLD_FAN_IN = 18.0  # the in-degree THRESHOLD is quoted at: an interior hex cell's two rings at REACH 2 (AUTHORITY.md
# §5.2, Byron, September 14, 2026). A container that scales starts neuron j at THRESHOLD * d_j / THRESHOLD_FAN_IN, so a
# neuron wired like that cell keeps 0.25 exactly and goo's 79 incoming synapses ask proportionally more. Goo scales;
# nothing else does yet, because turning it on for the grid would move every threshold every result was measured at.
MINIMUM_POTENTIAL = -1.0  # floor on a potential: inhibition and carried-over charge can go no lower

# --- goo, the working network (AUTHORITY.md §3.4; Byron, September 14, 2026: "We will speed everything up by
# selecting 60 units of goo, with THRESHOLD=1") -----------------------------------------------------------------
GOO_COUNT = 60  # neurons in goo when --goo is given no number: 3,540 connections against 80's 6,320, about twice the speed
GOO_THRESHOLD = 0.2  # goo's THRESHOLD, quoted per THRESHOLD_FAN_IN like the grid's and scaled by goo's fan-in (§5.2): a goo
# of 60 starts at 0.2 * 59/18 = 0.66. Set from the fine sweep of §3.4 (Byron, September 14, 2026, "the word"): with every
# neuron un-sticking, 0.15-0.40 is a plateau and 0.20 the one level where every seed learned. It was 1 -- theta 3.28,
# chosen to sit past the saturation edge -- which turned out to be off the plateau: a dead interior on the no-direct
# copy. The grid keeps 0.25 -- the threshold belongs to the container, the third reading §3.4 named, adopted for goo
GOO_MINIMUM_POTENTIAL = GOO_THRESHOLD * MINIMUM_POTENTIAL / THRESHOLD  # the floor follows at the grid's ratio of -4, as
# every goo sweep ran it (§5.2, one axis, two points)
GOO_SCALING_FACTOR = 0.05  # goo's wiring, the scaled rule (AUTHORITY.md §3.4; Byron, September 16, 2026: "P(i projects
# onto j) = 0 if i == j; 0 if i and j are both in the input zone; P_ij necessary to give j an average of N * scaling_factor
# inputs. Please default scaling_factor to 0.05"). Every neuron hears N times this many synapses in expectation -- 32.2 on
# the mnist goo of 644, 3 on goo 60 -- at the probability that fan-in makes over the sources it may hear (N - 1 outside the
# input zone, N - I inside it), stopped at 1. "I realize this does not give like-for-like comparisons, but that's OK
# because we aren't going to be comparing to an oversaturated or dull network"
GOO_PROJECTION = 0.2  # the probability of the three earlier wirings (--wiring zones-equal, zones, uniform), superseded as
# the wiring's knob by GOO_SCALING_FACTOR on September 16, 2026 (§3.4). Under the zone rule: P(neuron i projects onto
# neuron j) for a pair with an interior end (Byron, September 14, 2026: "P(i connects to j) = 0 if i == j; 0 if i in
# inputs or outputs AND j in inputs or outputs; P_connection otherwise" -- and, correcting the verb, "I should have said
# projects. The connections are all one-way"). Pairs with both ends in a zone never project, and an interior-to-zone
# projection is scaled up so every neuron hears the same number in expectation; below 1 the seed decides the wiring.
# Set from Byron's two sweeps of September 14-15 (§3.4): the plateau in P runs 0.15 to 0.5 with cliffs at 0.1 and from
# 0.6 up, and 0.2 sits inside it with every seed learning on either side, the highest floor anywhere, and the fastest goo
# that learns -- about 650 projections at sixty neurons, four times the speed of the fully connected goo, which was 1
TAU = 2.0  # ms: leak time constant of the potential, computed lazily on arrival (Byron, September 12, 2026, bringing the leak back; his earlier sweep chose 2); math.inf switches it off
REFRACTORY = 5.0  # absolute refractory period: a neuron that fired this recently ignores every signal
REFRACTORY_HOPS = 3.0  # the refractory period divided by the time a signal takes to travel one hop; not an integer (Byron, September 11, 2026)
INTERVAL = 35.0  # ms: the epoch's length, the spacing of inputs when no time is given. Swept September 14, 2026 on
# shallow_copy over 5 to 45 ms: the optimum is a plateau at 35-40 and 35 is the cheaper of the two, against the 20 ms
# the problems had inherited and never chosen (Byron, same day, defaulting it here and removing every override)
BORED_AFTER = 0.0  # off (Byron, September 14, 2026). When positive it is the ms of silence after which a neuron's
# threshold has fallen to zero and it fires on its own (§5.4, Byron, September 12, 2026). Swept below the epoch it
# floods: at 10 ms the output row fires 96.5% of the time whatever the input, and copy falls from 0.838 to 0.542.
# Superseded by ESCAPE_DELTA (Byron, September 15, 2026: "The hazard is buying us what the bored clock was supposed to
# buy us, and much much more cleanly"): not run on top of the hazard; the mechanism stays, the configuration does not.
ESCAPE_DELTA = 0.455  # the firing decision is a draw (AUTHORITY.md §5.2, escape noise; Byron, September 15, 2026:
# "Make the boredom stochastic and it is Williams's unit outright"): a neuron that is not refractory fires at a wave
# with probability 1 - exp(-m), m = (dt / hop) * exp(s / delta_j), s its margin p - theta(t) and delta_j this constant
# times its starting threshold -- one expected spike per hop at threshold, e times more per delta_j above it. 0 is the
# deterministic threshold. The value is Byron's word (September 15, 2026, after the Delta x LR grid of §3.4): inside the
# plateau that runs 0.25 to 0.7, where every seed learns at LR 0.02 and up. Applied by the command line and the sweep
# driver (--delta); a network built in the library is deterministic until Network.set_delta, as it has no noise until a
# Teacher gives it sigma. The hazard eligibility (§6.7) needs it.

# --- how a bit becomes spikes (AUTHORITY.md §4.3) ---------------------------------
INPUT_DRIVE = "rate"  # "rate": a Poisson process DRIVES each input neuron across the epoch, each arrival at its own
# continuous time. "forced": every bit-1 neuron is made to spike at once at the epoch's moment, which locks every
# spike in the network onto a hop grid anchored there -- a unified wave front at time zero, which Byron ruled out on
# September 14, 2026: "I don't want any such thing."
INPUT_CV = 0.6  # the drive is specified by the coefficient of variation of the spike train it produces, not by its
# own rate (Byron, September 14, 2026). Arrivals inside the refractory period are dropped, so the neuron fires at the
# first arrival after it ends and its spike train is a renewal process with DEAD TIME, not a Poisson one: mean ISI =
# REFRACTORY + 1/lambda. That ties the rate and the CV together exactly -- the train runs at (1 - CV)/REFRACTORY, so
# 200 * (1 - CV) Hz -- and CV is the end of that axis worth naming, because it is the quantity the neuroscience
# literature reports. 0.6 gives lambda 0.133/ms, a 12.5 ms interval, 80 Hz and 2.8 spikes across a 35 ms epoch. It
# sits inside the 0.5-1.0 that visual cortex shows (Softky & Koch 1993) and well clear of the starvation at CV 0.95,
# where an epoch holds a third of a spike. The sweep of September 14 found the whole middle of this axis flat, so the
# choice rests on the biology, not on the measurement.


def rate_for_cv(cv: float, refractory: float = REFRACTORY) -> float:
    """The drive rate lambda whose spike train has this coefficient of variation (AUTHORITY.md §4.3).

    Needs a refractory period: the dead time is the only thing that makes the train sub-Poisson, so with
    REFRACTORY at 0 the CV is 1 at every rate and no lambda answers a request for less.
    """
    if not 0.0 < cv < 1.0:
        raise ValueError(f"CV must be strictly between 0 and 1 (0 needs infinite drive, 1 needs none), got {cv}")
    if refractory <= 0.0:
        raise ValueError("a CV below 1 needs a refractory period; with none the drive is Poisson whatever its rate")
    return (1.0 - cv) / (refractory * cv)


def cv_for_rate(rate: float, refractory: float = REFRACTORY) -> float:
    """The coefficient of variation of the spike train a drive rate of lambda produces; 1.0 for no drive at all."""
    return 1.0 if rate <= 0.0 else (1.0 / rate) / (refractory + 1.0 / rate)


INPUT_RATE = rate_for_cv(INPUT_CV)  # per ms: the rate of the driving process, not the rate the neuron fires at
INPUT_RATE_OFF = 0.0  # per ms: the driving rate for a bit-0 neuron; 0 makes a zero bit mean silence, as forced drive does

# --- reading a rate rather than a bit (AUTHORITY.md §4.3, §6.9) --------------------
RATE_TAU = 5.0  # ms: the exponential window the read estimates a firing rate over (Byron, September 14, 2026). Each
# spike puts 1/RATE_TAU on the neuron's trace and it decays with the same constant, so no spikes means a rate of zero.
RATE_ON = 200.0  # Hz: the rate an output the target says should be on is driven to. 200 Hz is 1/REFRACTORY, the fastest
# the absolute refractory period allows: FOR NOW the teacher aims at saturation, not at a middling set point (§6.9).
RATE_OFF = 0.0  # Hz: and one that should be off is driven to silence.
TEACHER_THRESHOLD = 14.3  # Hz: the "count" read (AUTHORITY.md §4.3; Byron, September 14, 2026: "COUNT the number of
# times each neuron fired in the epoch. ESTIMATE the firing rate based on the count. If the firing rate estimate exceeds
# TEACHER_THRESHOLD, the output neuron is 1. Otherwise it is zero"). The rate is the epoch's count over its length, so
# at 35 ms one spike is 28.6 Hz: 14.3 is the middle of the one-spike band, halfway between no spike and one, so an
# output is on if it fired at all this epoch and the line sits as far from both edges as it can (Byron, the same day,
# setting it to mean one spike -- not for the score it yields; the sweep of §4.3 is a measurement of one synapse). It
# was 40, two spikes, from the read's first hour.
READ_WINDOW = 5.0  # ms: the window of the "window" read -- a bit, but only counting spikes this recently before the
# epoch's end (Byron, September 14, 2026, going forward with bit reading and a five-millisecond window)

# --- the problem ------------------------------------------------------------------
PROBLEM = "reversal"  # what the network is asked to do and how it is watched (problems.PROBLEMS)

# --- learning: dopamine (AUTHORITY.md §6) --------------------------------------------
RULE = "teacher"  # which learning rule runs (learning.RULES): teacher (an external teacher scores the read, Byron, September 12, 2026),
# dopamine (the student as its own teacher), or the reinforce rule factored out below
TEACHER_CREDIT = None  # credit per neuron in the teacher's score; None normalises it to span [-1, 1] whatever the zone's size
# (Byron's 0.25 is 1/4, the four-input zone; twelve outputs give 1/12, so six right is zero)
POPULATION = 3  # neurons per raw bit under population coding (Byron, September 13, 2026): 1001 -> 111000000111
FLIP = 1.0 / 12.0  # the probability a problem that corrupts its input flips each coded bit with (Byron, September 13, 2026,
# reading the input zone back): a network built in the library does not flip until it is asked to; 0 = off

# --- quashing cycles (AUTHORITY.md §6.11) -------------------------------------------
QUASH_RATE = 0.02  # the rate a problem that quashes uses: a refire weakens each contributing synapse by this fraction
# of its weight. A network built in the library does not quash until it is asked to; 0 = off.
QUASH_K = 0.2  # per ms: the quash falls off as exp(-k * (t - t_fired)) with the delay since the previous spike

# --- leaky Hebb (AUTHORITY.md §6.12) -------------------------------------------------
SYNAPSE_TAU = 10.0  # ms: the leak of the eligibility trace on a synapse (§6.12). It was taken equal to the neuron's TAU
# 'for computational simplicity', which bought an identity and broke the rule: at TAU 2 the trace attenuates 148x across
# the 10 ms a network computes over, and every value from 5 ms up recovers it (Byron, September 13, 2026, deciding to
# decouple them). It governs leaky_hebb and the reinforce rule's leaky eligibility alike.
LEAKY_ELIGIBILITY = False  # append the leaky trace of §6.12 to the reinforce rule's chain, so the global reward reaches
# each synapse in proportion to what it was still contributing (Byron, September 13, 2026); off keeps the pre-alpha's rule
HEBB_RATE = 0.01  # the rate a problem that runs leaky_hebb uses: a firing neuron potentiates each synapse that still
# had charge in it by this much times the synapse's leaky trace (Byron, September 13, 2026). A starting value, to be
# swept. Like the quash it composes with whatever else runs, and a network built in the library leaves it off; 0 = off.
LR = 0.03  # learning rate, both rules
SIGMA = 0.1  # exploration noise: std dev added to each neuron's potential; 0 switches it off
EXPLORE = "wave"  # when that draw is taken (AUTHORITY.md §6.1): "wave", afresh before every firing decision, so a
# neuron's xi is the perturbation it actually decided under (Byron, September 13, 2026), or "epoch", once at the
# input's moment, which is the pre-alpha's and what §6.7's baseline was measured with
DOPAMINE_RELEASE_ALPHA = 2.0  # shape of the gamma density of the amount a refire releases against its delay past the refractory period (Byron, September 12, 2026)
DOPAMINE_RELEASE_THETA = 1.0  # ms: its scale; the release peaks at (alpha - 1) * theta past the end of the refractory period
DOPAMINE_TAU = 20.0  # ms: decay of the global dopamine value
DOPAMINE_EXPECTATION_TAU = 600_000.0  # ms (10 minutes): the exponential window of the expected dopamine trace (Byron, September 12, 2026)
DOPAMINE_EXPECTATION_START = 0.0  # where the expected dopamine trace starts; a high start holds early learning back (Byron, same day)
DOPAMINE_ORDER = "release-first"  # at a refire, release before the weight update, or update-first (dopamine.ORDERS)
DOPAMINE_PUNISH = True  # an input neuron whose bit is 0 has the sign of its update reversed when it refires (Byron, September 12, 2026)
DOPAMINE_PUNISH_GAIN = 2.0  # and that reversed update is this many times as large as a reward (Byron, September 12, 2026, 'for now')
WEIGHT_DECAY = 1e-4  # every weight moves toward 0 by this fraction each epoch: synapses that forget on their own (Byron, same day, 'for now')

# --- the reinforce rule of the pre-alpha, factored out behind RULE = "reinforce" ---
TARGET = "reversed"  # what the top row should show, derived from the input row (learning.TARGETS)
CRITIC = "row"  # how the reward is judged (learning.CRITICS)
TEMPERATURE = 2.0  # the evidence critic's temperature (AUTHORITY.md §8; Byron, September 16, 2026: "the spikes are EVIDENCE"):
# the class sums are read as log-odds at this scale, q_k = exp(n_k / T) / sum_j exp(n_j / T), and the reward is ln q_y; a lead of
# T spikes makes a class e times as likely. 0 would be the class critic, infinity a flat ln 0.1. Set at the middle of the first
# sweep, {1, 2, 4}; "We will have to sweep for temperature eventually"
ELIGIBILITY = "perturb"  # what the global reward acts on when the threshold decides (learning.ELIGIBILITIES): the
# pre-alpha's. The Teacher and the command line take "hazard" instead on a network with escape noise (ESCAPE_DELTA > 0,
# §5.2) unless told otherwise -- the eligibility every measurement at 0.455 was made with; additive noise on top of the
# hazard was never measured (Claude's reading of the default Byron set, September 15, 2026)
LATE = "count"  # what a signal arriving after its target fired earns (learning.LATE_RULES)
BASELINE_RATE = 0.05  # per-epoch update of the running reward baseline the advantage is measured against
WINDOW = 200  # epochs the Teacher's moving-average accuracy spans
HOMEOSTASIS = 1e-6  # per-epoch rate at which a threshold moves toward the target firing rate; 0 = off
TARGET_RATE = 0.5  # firing rate homeostasis aims for, 0 to 1
UNSTICK = 1e-3  # per-epoch rate at which a stuck neuron's threshold moves toward UNSTICK_TARGET; 0 = off. Every neuron,
# not the output row only, since September 14, 2026 (AUTHORITY.md §6.7): the interior of a goo with no direct
# projection was dead for want of it, and 'all neurons are first-class citizens' (Byron)
UNSTICK_TARGET = 0.5  # firing rate the un-sticking aims for
# THRESHOLD_RANGE, the [-5, 5] homeostasis and un-sticking clipped thresholds to, was eliminated on September 14, 2026
# (Byron: "It's artificial"; AUTHORITY.md §1.3, §3.4). A threshold goes where the rules take it.
RATE_MEMORY = 0.01  # per-epoch update of a neuron's running firing rate (about the last 100 epochs)
STUCK_BELOW, STUCK_ABOVE = 0.01, 0.99  # a neuron firing less or more often than this is "stuck"
