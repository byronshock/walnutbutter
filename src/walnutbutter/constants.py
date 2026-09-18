"""Every global constant of walnutbutter, in one place.

These are the defaults the constructors, the neuron's clock, the Teacher and
the command line all read from: change a value here and every path follows.
They are the physics of the substance (what a neuron needs to fire, how fast
it leaks, how learning moves a weight) and the shape of the default network.
The format fallbacks for old checkpoints stay in persistence.py, because they
record what those files meant when they were written, not what the default is
now.

The clock values live here but run from `Neuron.refractory` and
`Neuron.refractory_hops`, the class attributes the command line sets (and
restores) per run.
"""

# --- the default network: footprint and wiring ----------------------------------
ACROSS = 8  # the input zone's width: the first ACROSS neurons, one per coded bit (§4.3)
WEIGHT_RANGE = (-1.0, 1.0)  # random weights are drawn from this range, and learning clips to it
WEIGHT_EPSILON = 0.001  # --epsilon: the smallest weight allowed under --positive-weights, range (epsilon, 1)

# --- the neuron: activation and the clock (nominal milliseconds) ----------------
THRESHOLD = 0.25  # total weighted input a neuron needs before it fires, quoted at THRESHOLD_FAN_IN incoming synapses
THRESHOLD_FAN_IN = 18.0  # the in-degree THRESHOLD is quoted at (AUTHORITY.md §4.11, Byron, September 14, 2026): the
# eighteen an interior cell of the archived hex grid heard, kept as the unit although that container has left the
# specification. A container that scales starts neuron j at THRESHOLD * d_j / THRESHOLD_FAN_IN.
MINIMUM_POTENTIAL = -1.0  # floor on a potential: inhibition and carried-over charge can go no lower

# --- goo, the working network (AUTHORITY.md §3.4; Byron, September 14, 2026: "We will speed everything up by
# selecting 60 units of goo, with THRESHOLD=1") -----------------------------------------------------------------
GOO_COUNT = 60  # neurons in goo when no problem and no --goo names a count: 3,540 connections, the working network
GOO_THRESHOLD = 0.2  # goo's THRESHOLD, quoted per THRESHOLD_FAN_IN and scaled by goo's fan-in (§5.2): a goo
# of 60 starts at 0.2 * 59/18 = 0.66. Set from the fine sweep of §3.4 (Byron, September 14, 2026, "the word"): with every
# neuron un-sticking, 0.15-0.40 is a plateau and 0.20 the one level where every seed learned. It was 1 -- theta 3.28,
# chosen to sit past the saturation edge -- which turned out to be off the plateau: a dead interior on the no-direct
# copy. The threshold belongs to the container, the third reading §3.4 named, adopted for goo
GOO_MINIMUM_POTENTIAL = GOO_THRESHOLD * MINIMUM_POTENTIAL / THRESHOLD  # the floor follows at the ratio of -4, as
# every goo sweep ran it (§5.2, one axis, two points)
GOO_SCALING_FACTOR = 0.05  # goo's wiring, the scaled rule (AUTHORITY.md §3.4; Byron, September 16, 2026: "P(i projects
# onto j) = 0 if i == j; 0 if i and j are both in the input zone; P_ij necessary to give j an average of N * scaling_factor
# inputs. Please default scaling_factor to 0.05"). Every neuron hears N times this many synapses in expectation -- 32.2 on
# the mnist goo of 644, 3 on goo 60 -- at the probability that fan-in makes over the sources it may hear, stopped at 1: the
# hidden neurons for an input, the inputs and hidden neurons for an output, everyone else for a hidden neuron, since the
# outputs were kept apart the same night ("With hidden=0 we have no cycles ... a two-layer feedforward network"). "I
# realize this does not give like-for-like comparisons, but that's OK because we aren't going to be comparing to an
# oversaturated or dull network"
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
REFRACTORY_HOPS = 2.0  # the refractory period divided by the time a signal takes to travel one hop; not an integer (Byron, September 11,
# 2026). 3 until September 17, 2026 (Byron: "Please set hops=2 by default"): a hop of 2.5 ms, not 1.67
INTERVAL = 35.0  # ms: the epoch's length, the spacing of inputs when no time is given. Swept September 14, 2026 on
# shallow_copy over 5 to 45 ms: the optimum is a plateau at 35-40 and 35 is the cheaper of the two, against the 20 ms
# the problems had inherited and never chosen (Byron, same day, defaulting it here and removing every override)
BORED_AFTER = 0.0  # off (Byron, September 14, 2026). When positive it is the ms of silence after which a neuron's
# threshold has fallen to zero and it fires on its own (§5.4, Byron, September 12, 2026). Swept below the epoch it
# floods: at 10 ms the output row fires 96.5% of the time whatever the input, and copy falls from 0.838 to 0.542.
# Superseded by ESCAPE_DELTA (Byron, September 15, 2026: "The hazard is buying us what the bored clock was supposed to
# buy us, and much much more cleanly"): not run on top of the hazard; the mechanism stays, the configuration does not.
ESCAPE_REFERENCE_COUNT = 60  # the count ESCAPE_DELTA is quoted at (AUTHORITY.md §5.2; Byron, September 16, 2026: "scaling
# the network MUST reduce the probability of escape noise at each neuron by sqrt(N)"): a network of N neurons runs every
# hazard at sqrt(this / N) times what the width alone gives, so the goo of 60 the width was set on keeps its regime and a
# larger network is quieter as the square root of its size. The unit the width is quoted in, as THRESHOLD_FAN_IN is the
# unit the threshold is quoted in; not a knob.
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
ROW_CRITIC_PICKINESS_IN_SPIKES = 2  # the count read's line, in spikes (AUTHORITY.md §5.10, §9.5): an output neuron is
# **on** for the row critic when its count for the epoch is at least this, an integer. It replaced the 14.3 Hz rate line
# on September 17, 2026, so the read no longer changes meaning with the epoch's length (Byron: "The row critic needs a
# parameter, ROW_CRITIC_PICKINESS_IN_SPIKES. It should be an integer, probably 1 or 2", and "An epoch is going to be
# 35 ms at pickiness 2"). Two is chosen against the escape hazard's rest rate: lower reads background as signal, and it
# travels with INTERVAL 35 ms rather than alone (§9.5)
READ_WINDOW = 5.0  # ms: the window of the "window" read -- a bit, but only counting spikes this recently before the
# epoch's end (Byron, September 14, 2026, going forward with bit reading and a five-millisecond window)

# --- the problem ------------------------------------------------------------------
PROBLEM = "reversal"  # what the network is asked to do and how it is watched (problems.PROBLEMS)

# --- learning: the rule that pays at the read (AUTHORITY.md §9.1) --------------------
RULE = "local"  # which rule pays at the read (learning.RULES): "reinforce", the one rule the specification carries,
# or "local" for none -- §9.1, "a run may have none, in which case the local rules are the whole of the learning".
# Every problem the specification carries names its own, so this is the library's default and not a run's
POPULATION = 3  # neurons per raw bit under population coding (Byron, September 13, 2026): 1001 -> 111000000111
FLIP = 1.0 / 12.0  # the probability a problem that corrupts its input flips each coded bit with (Byron, September 13, 2026,
# reading the input zone back): a network built in the library does not flip until it is asked to; 0 = off

# --- quashing cycles (AUTHORITY.md §6.11) -------------------------------------------
QUASH_RATE = 0.02  # the rate a problem that quashes uses: a refire weakens each contributing synapse by this fraction
# of its weight. A network built in the library does not quash until it is asked to; 0 = off.
QUASH_K = 0.2  # per ms: the quash falls off as exp(-k * (t - t_fired)) with the delay since the previous spike

LR = 0.03  # learning rate, both rules

# --- the reinforce rule of the pre-alpha, factored out behind RULE = "reinforce" ---
TARGET = "reversed"  # what the output zone should show, derived from the input zone (learning.TARGETS)
CRITIC = "row"  # how the reward is judged (learning.CRITICS)
TEMPERATURE = 2.0  # the evidence critic's temperature (AUTHORITY.md §8; Byron, September 16, 2026: "the spikes are EVIDENCE"):
# the class sums are read as log-odds at this scale, q_k = exp(n_k / T) / sum_j exp(n_j / T), and the reward is ln q_y; a lead of
# T spikes makes a class e times as likely. 0 would be the class critic, infinity a flat ln 0.1. Set at the middle of the first
# sweep, {1, 2, 4}; "We will have to sweep for temperature eventually"
ELIGIBILITY = "hazard"  # which of the two eligibilities of AUTHORITY.md §8.3 a run gets when it names none. Both are
# the single-spike rule of §8.4 and differ in what a decision's credit and expectation are: hazard takes the escape
# decision's own score (§8.7), which is exactly zero-mean at every decision; hebb takes the neuron's own estimate of its
# spike (§8.6), whose mean is the lag of that estimate, so a run under hebb posts a systematic component wherever a
# neuron's rate is moving. Where the threshold decides, no eligibility runs and the rule refuses to learn (§8.3)
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
DECISION_MEMORY = 1e-4  # per-decision update of a neuron's expectation of its own spike, p_hat_j, which the hebb eligibility
# charges at every decision (AUTHORITY.md §6.7, the single-spike rule; Byron, September 17, 2026: "Expectation is changed
# per decision in this architecture"): about the last 10,000 decisions. Every neuron decides at every wave and every
# Poisson arrival of the drive is a wave, so on the mnist feedforward goo at 100 ms a neuron makes about 3,300 decisions
# an epoch (measured September 17, 2026): the window is about three epochs, not the 170 a wave a hop would give. Until
# that many decisions have been seen the estimate is their plain mean (a rate of 1/n at the n-th), and the first decision
# sets it and charges nothing. A starting value, to be swept.
TARGET_ISI = 5.1  # ms: the interspike interval the ISI factor pays most for (AUTHORITY.md §0.2; Byron, September 17, 2026:
# "The desired ISI is 5.1 ms (hardcode for now)"). Not known: 0.1 ms past REFRACTORY for now
ISI_FACTOR = True  # weigh every charge of the single-spike rule (hebb and hazard, §6.7) by f(t - TARGET_ISI), t the time
# since the neuron's own last spike, f = (3x - 1) / (1 + x^3) at x = t / TARGET_ISI (§0.2); on by default (Byron, September
# 17, 2026), and a resumed network keeps the setting it was saved under
STUCK_BELOW, STUCK_ABOVE = 0.01, 0.99  # a neuron firing less or more often than this is "stuck"
