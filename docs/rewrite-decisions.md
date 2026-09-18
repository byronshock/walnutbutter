# The rewrite: the decisions only Byron and Cedric can make (September 17, 2026)

*From the reading of `RECORD.md` (see `rewrite-outline.md`). Ordered so that the ones blocking the most come first. Nothing below is decided.*

## 1. What fires a neuron, once θ is deprecated?

**Why it matters.** Firing is the event every other rule is defined on — the refractory period, the eligibility, the read, the teacher's rate error. The one section marked non-negotiable currently defines it by a threshold that no longer exists, so nothing below it can be written, and because the new file's rule is that a clause is tested before it is built, no test can be written either. Your own phrase, "random walks that may or may not hit the threshold", leaves open whether a level survives under another name.

**The choices.** (a) The synapse hazard alone: a neuron spikes only when one of its synapses takes its exploration event, and the potential causes nothing — the most literal reading of exploration at the synapse, and it makes the potential a pure record of evidence the teacher reads. Cost: the accumulator's evidence drives nothing. (b) The potential sets a hazard rate with no threshold parameter (a rate monotone in p, quoted per hop) — keeps Williams's unit and lets evidence cause firing, but a rate needs a scale, and a scale quoted per fan-in is θ's fan-in scaling under another name. (c) A level that is not called θ — cheapest to build, but it is θ renamed unless the level is fixed and never rescaled by fan-in or container.

**Blocks.** §0.2's mechanism, §6.1, all of §7, and the activation code and its first test in every engine. Everything else waits on it.

## 2. Does the potential reset when a neuron fires, and is there a floor?

**Why it matters.** The reset was never specified on its own — it rode in the tail of the threshold sentence ("at which point it resets"), and that sentence is going. With no leak, no threshold and possibly no floor, the potential is an unbounded random walk: a neuron that drifts down is silent forever and one that drifts up fires on every draw. The floor was quoted against θ (the grid's −4 ratio), so deprecating θ takes the floor's unit away even if you keep a floor. The floor is also where dp/dw = x_ij stops being exact, so it is a fact about the eligibility and not only about the potential.

**The choices.** Reset to zero, reset by subtracting the level crossed, or no reset at all; and a floor, or none — your words: "Maybe there is no floor and we learn at first with random walks that may or may not hit the threshold." No reset and no floor is the purest accumulator and the likeliest to produce dead neurons; a reset is the only bound left if there is no floor; a floor without a threshold needs a new unit to be quoted in (an absolute value, or a multiple of the mean weight, or per fan-in).

**Blocks.** §1.4, §1.5, the neuron's state and the checkpoint format, the arrival and per-wave steps in every engine, and any claim about what a network does when started from silence.

## 3. What exactly is the synapse hazard — per synapse or per neuron, per hop or per refractory period, at what rate, and does the √N law survive it?

**Why it matters.** Under exploration at the synapse this is the only thing that makes anything fire, so its constant sets the whole system's activity. The record states it in two units that disagree by about a factor of two: (2d−1)/(2d²) per refractory period in §0.4, and "exactly 1/d within a hop" in §0.3 — at REFRACTORY_HOPS = 2 a refractory period is two hops. Deeper: it is stated per neuron but motivated per synapse, and d independent synapse draws do not give a neuron-level union of that form exactly. And it is the rule that decides whether §4's wirings must equalise fan-in at all: equal fan-in used to matter because θ scaled with it, and that reason is gone — equal exploration rate would be a new and arguably better one.

**The choices.** On the unit: say which of the two statements is the definition and which the consequence. On the form: (a) per synapse, each drawing for itself, the neuron-level rate a consequence; (b) per neuron, d only an argument. On the rate: keep (2d−1)/(2d²) "FOR NOW" — no free parameter, perfect for "as few knobs as possible", and an output of fan-in 395 explores about once every 2 s; or set it from a perceptual frame, about 5 ms/T_f per refractory period (0.05 at 100 ms), one constant that reaches your goal but no longer falls with d at all. A frame-set rate interacts with the ISI factor: a charge made a frame after the last spike is weighed at 0.030 (50 ms), 0.0077 (100 ms) or 0.0019 (200 ms), so exploration on the frame scale earns almost nothing under f as built. On √N: say whether "scaling the network MUST reduce the probability of escape noise at each neuron by sqrt(N)" is discharged by a fan-in form (which already falls as the network grows on a dense container) or is still owed on top of it — keeping both scales exploration down twice.

**Blocks.** §7.2 and §7.4, the draw order every engine must share, the sizing of every network, §4.11 (what equal fan-in is for), and every sweep — a hazard rate that moves moves every number measured under it.

## 4. What happens to a synapse's exploration event when its target neuron is refractory?

**Why it matters.** §0's third bullet says a refractory neuron ignores its inputs and does not integrate them, which answers it for arrivals. A hazard event is not an arrival. Quoting the hazard "per refractory period" hints the hazard may not run at all on a refractory neuron. The three readings give measurably different exploration rates, and they diverge on the first wave — so the engines cannot be written before it is settled, because the agreement tests would fail with no bug to fix.

**The choices.** Dropped (the bullet's reading, and exploration is lost at a rate that rises with the hazard); queued until the neuron returns (breaks "ignores its inputs"); or the hazard simply does not run while the neuron is refractory, which is what makes "per refractory period" the natural unit.

**Blocks.** §1.6 and §7.3 together, and the wave loop in every engine.

## 5. Which eligibility survives?

**Why it matters.** Your own open question, and the single largest driver of the rebuild's size: each surviving eligibility is a separate implementation in every engine, held to the bit, plus its own constants (LEAKY_ELIGIBILITY, SYNAPSE_TAU, HEBB_RATE, COUNT_MEMORY, DECISION_MEMORY). The current default is `hazard` on a network with escape noise, and that default is Claude's reading rather than your word — and `hazard` is defined on a margin p − θ that θ's deprecation removes.

**The choices.** hebb, the single-spike per-decision rule of September 17, which is built, checkpointed, and the one the ISI factor was written for — the obvious survivor, and the only one that still has a definition once θ is gone. Against it: perturb never moved off chance and its noise source is superseded twice; wrong_hebb carries no digit; count_hebb is the epoch form the single-spike rule replaced; hazard needs a margin. Also to settle inside hebb: whether its credit at the spike is a full unit or discounted by the expectation — you deferred that on September 17 ("Defer for now"), and it is still deferred. And whether a local rule composes on top (the quash, weight decay, a leaky Hebb) or the rewrite runs one rule and nothing else.

**Blocks.** All of §8, and therefore the clause-before-code order for the whole learning section. Also §8.5 (DECISION_MEMORY's window) and §8.18–8.19.

## 6. What survives of §0's dopamine bullet — is there still one global scalar reward, produced locally at a spike and consumed globally?

**Why it matters.** It is one of four non-negotiable statements and you have said it no longer describes the system. The timing law is certainly dead: it says dopamine is maximal immediately after the refractory period and decays, where f is negative at t = 0 and peaks at TARGET_ISI — the opposite shape. What is unsettled is whether the frame around it survives, and that frame is the §0-level form of the eligibility question.

**The choices.** Keep the frame and drop the timing, letting f supply the timing the bullet used to state — one sentence, and §0 stays honest. Or keep neither, and let the teacher's rate error be the whole learning signal with no global scalar at all — which makes the whole of §6.2–§6.6 dead code the rewrite must not carry. Or keep "dopamine" only as the name of the global term A that multiplies the eligibility — which keeps a word whose mechanism has gone, and that is how this bullet drifted in the first place.

**Blocks.** §0.6, and every learning clause that names a global signal. The eight dopamine constants and their Rust gap go with it.

## 7. The new teacher: its name, its observation window, its critic, and when it pays.

**Why it matters.** It is the largest undecided thing in the file, and seven constants wait on it (RATE_TAU, RATE_MEMORY, TEACHER_THRESHOLD, READ_WINDOW, RATE_ON/RATE_OFF, TEACHER_CREDIT, CRITIC). The memory rule applies with full force: say what "on" means and measure what the network produces before any sweep — a read that counts background as signal makes every sweep on it wrong.

**The choices.** The window has three candidates already in the constants: RATE_TAU's 5 ms (one interspike interval, a rate read off a single spike), RATE_MEMORY's roughly 100 epochs, or the epoch itself. The targets are settled in direction — the drive rate for a 1 (one spike every 5.1 ms, 196 Hz) and the exploration rate for a 0, which is new, non-zero, and answers the risk you asked to be recorded on September 14, that the old targets were both extremes with no set point between them. Note the coupling this creates: a 0-target equal to the exploration rate makes the teacher depend on the hazard's constant, a new link between §7 and §9. The critic is open across six existing ones plus whatever a rate difference implies. And whether the teacher pays once at the horizon or continuously: a teacher that charges per decision has no single reading instant, and if it charges continuously the epoch stops being a learning unit at all.

**Blocks.** §9 entirely, §1.8, §5.9, §3.12, and any measurement on the rebuilt system.

## 8. Does the ISI factor gate the teacher only, or every charge of the eligibility — and is t the time since the neuron's own last spike?

**Why it matters.** Your sentence was "the teacher's existing reinforcement is MULTIPLIED by f(t-ISI)", which reads as the teacher alone. What was built weighs every charge of the single-spike rule, credit and expectation alike, so a neuron's silence is charged against it at every decision, weighted by how long it has been silent. The record marks that as Claude's reading, "to be corrected in a word", and it is running by default in the frozen code — so every result from here is measured under whichever reading stands. Also unsettled: what reaches a neuron that has never fired, where f = 0 and nothing at all is charged. That was harmless while θ and the escape margin guaranteed a first spike; with no threshold and maybe no floor, a network can start in silence and this clause says it cannot be taught out of it.

**The choices.** The teacher only (narrowest, matches your words); every charge of whichever eligibility survives (what is built and tested); or both, named as two clauses so they can differ later. For t: since the neuron's own last spike (built), since the presynaptic spike, or since the last arrival at the synapse — a real choice now that exploration is a synapse's event. For the never-fired neuron: the hazard is the whole bootstrap and the teacher waits for the first spike; or a floor value of f so the teacher can reach it; or the rate error acts outside the gate entirely, which is close to saying the factor gates the eligibility and not the teacher.

**Blocks.** §8.12, §8.13, §9.2, and the never-fired branch in every engine.

## 9. Which containers and problems survive — goo alone, or goo plus the grid; goo and mnist only?

**Why it matters.** The substance is defined as spread on a plane, and goo is the plane taken away — yet goo is the working network and every September result, mnist included, is on it. Roughly a third of §2 and all of §3.1–§3.2 turn on this, plus eight constants (ACROSS-as-cells, ROWS, OMEGA, REACH, POPULATION, FLIP, TARGET, PROBLEM). If the grid goes, the substance needs a definition that does not mention the plane. If it stays, it stays as a question the project has not answered: both sweeps that asked what locality and depth are worth ran under a rule that did not learn.

**The choices.** (a) Goo only: §3 becomes one connection clause plus the goo wirings; the shortcut rejection loop — the fiddliest cross-engine stream-order requirement in the file — disappears. Cost: the locality question cannot be asked again without rebuilding a container. (b) Goo plus the grid: keeps the comparison alive at the cost of rebuilding positions, the hex metric, the neighbourhood and the shortcut draw in every engine. (c) All four: the columns and the lattice add little the grid does not and have no September result. On problems: goo and mnist only drops copy's grid siblings, the permutation and the reversed target in one stroke — all already artefacts by your first-class-citizens rule.

**Blocks.** §4.2, §4.14, §5.2, §10.2, eight constants, and the container code the rewrite carries over. The permutation's removal also changes what every seeded stream reproduces, so it must be decided before anything is rebuilt.

## 10. Which engines are built, and must every clause agree to the bit?

**Why it matters.** The file's rule is that every clause is in every engine and the engines are compared with ==. The ISI factor already states two tolerances: objects and Rust agree bit for bit, the array engine to a part in 1e9. A specification cannot say "==" and carry a floating tolerance without naming which clauses it covers. Under the new file the agreement tests are written before the code they test, so this is answered before the first engine is started.

**The choices.** All three engines, with the array engine's tolerance written in as a named exception where floating-point associativity makes bit agreement impossible; or objects plus Rust only, with == everywhere and no tolerance clause at all, and the array engine kept in the record as history — which is where your standing rules about sweeps on Rust and gaps closed in Rust already point. Related: where the constants live when Rust is first-class. Today the Rust loop takes every value from Python at construction, which makes Python a hard dependency of every run.

**Blocks.** §11.1, §11.4, §11.8, and the order in which the engines are written.

## 11. Under the new deterministic drive, what does a 0 bit do — and is TARGET_ISI one constant or three?

**Why it matters.** The replacement drive is one sentence and leaves two things unsaid. First, what happens on an undriven input neuron: silence is what `forced` and INPUT_RATE_OFF = 0 both meant, but the new teacher's target for a 0 is the exploration rate, so the drive and the target would then disagree about what a zero is. Second, TARGET_ISI has acquired three jobs — the factor's peak, the drive's period, and the teacher's target rate for a 1 — so a sweep on any one moves the other two. And the September 14 reason for dropping `forced` must be answered by the new drive: a single synchronised wave locks every spike onto one hop lattice.

**The choices.** A 0 bit means silence (consistent with the old drive, inconsistent with the new target); or a 0 bit is left to the synapse hazard, which makes drive and target agree. On the constant: one TARGET_ISI for all three jobs, stated in the clause as the same interval by intent — which keeps the network's returning spikes aligned with the drive, since at a 2.5 ms hop a two-hop return lands at 5.0 ms against a 5.1 ms target; or three constants sharing one default, which costs nothing now and stops a sweep silently moving the factor.

**Blocks.** §5.7, §5.8, §8.14, §9.3, the input-stream code carried over, and any sweep over the drive rate.

## 12. Does learning still clip every weight to [-1, 1], and what sets the scale of a weight now that θ is gone?

**Why it matters.** The weight clamp has never been examined because no sweep pressed against it, and it is the same shape of thing you struck down on the threshold axis on September 14 ("It's artificial"). It matters more now, not less: with no θ and no leak, the weight is the only scale left in the neuron, so [-1, 1] is the whole of the neuron's units rather than one scale among several. A hard clamp is also a place where a synapse stops responding to credit entirely, which sits badly with a synapse that is supposed to explore its own impulse response. And weight decay is the only force pulling back from the rails.

**The choices.** Keep the clip as the one bound, with weight decay as the counterweight; or remove it per the permissive-structures value and bound the network another way (a norm, a per-neuron budget, or a range scaling as 1/√N — a candidate the record never tested); or keep the range for the initial draw only and let learning go outside it, which is the minimal change and matches how the threshold is now treated. Separately: does the no-inhibition mode (`--positive-weights`, WEIGHT_EPSILON) survive at all?

**Blocks.** §2.5 and §8.15 — no update clause can be written without knowing whether it ends in a clip — plus §8.18.

## 13. Is INTERVAL still 35 ms, and is the epoch still the learning cadence?

**Why it matters.** The 35 ms was chosen on a sweep run under Poisson drive, where a longer epoch bought both settling time and more input spikes, and the record says outright that the two were never separated. Under a 5.1 ms deterministic clock an epoch holds a fixed number of drive spikes (6.9 at 35 ms, 19.6 at 100 ms), so the value rests on a measurement the new drive invalidates — and the mnist runs already use 100 ms. Meanwhile the single-spike rule already charges per decision, and you have said epochs "don't really exist in nature".

**The choices.** Keep 35 ms and the "no problem overrides it" rule, which puts the epoch's length in exactly one place; re-decide the value under the new drive; or make the epoch a reporting boundary only, with learning charged per decision, in which case INTERVAL stops being a learning constant at all — and WEIGHT_DECAY, BASELINE_RATE, RATE_MEMORY and COUNT_MEMORY, all quoted per epoch, need a new unit.

**Blocks.** §3.11, §3.12, §9.6, the schedule's horizon rule in every engine, and four per-epoch constants.

## 14. Does the learning rate have to fall now that the injected noise lives at the synapses?

**Why it matters.** The one result of Werfel, Xie and Seung the record keeps is that an estimator's largest usable learning rate falls as the first power of the dimension of the noise it injects. Moving exploration from neurons to synapses multiplies that dimension by the fan-in — 1,152 against 445 on the two-layer mnist goo, far more on a dense one — so the learning rates the recorded sweeps settled on were chosen under a different noise dimension and may be too large by that factor. The first sweep on the rebuilt engine would otherwise measure the wrong regime.

**The choices.** Leave the default where the neuron-noise sweeps put it and let the estimator average longer; scale the default down by the fan-in; or make it a swept constant quoted in a stated unit, the way the hazard's width used to be quoted at a starting threshold.

**Blocks.** §8.16's default value, and the first sweep on the rebuilt engine.

## 15. Does §0 stay non-negotiable, and do its bullets become dated, attributed clauses like every other?

**Why it matters.** Two of §0's four bullets are being rewritten and one was reversed twice in five days. A section marked non-negotiable that is nonetheless the most-revised part of the file teaches the opposite of what the marking intends, and the new file's own rule is that every clause is dated and attributed. Keeping §0 as it stands would carry a threshold and a dopamine timing law that no longer exist into the one section marked non-negotiable.

**The choices.** Keep §0 as a short list of design values that no clause may contradict — the inputs are the outputs, a neuron integrates deltas, the potential does not leak, the refractory period is a computational feature, exploration belongs to the synapse, structures are permissive, few knobs, the network keeps living — with every mechanism moved out into ordinary dated clauses (this is what the draft below assumes). Or drop the special status entirely and date every rule alike, relying on the file-wins-over-code rule for authority.

**Blocks.** The shape of §0, which every other clause's numbering hangs off, and the status of the four old bullets.

## 16. The small structural refusals: may a zone be any set of neurons, must the zones default to disjoint, and may a neuron connect to itself?

**Why it matters.** Each is small on its own and each is a refusal, and under the new file a refusal must be a clause rather than a convenience. Zones are "the first `across` neurons" and "the last `outputs`" because goo has no places — an index doing the work of a label, against the standing value that a neuron takes no role from its position or label. Self-connections were written into all four wiring rules by you and never argued; under a leaky neuron a self-synapse was nearly nothing, but under an accumulator it is a two-hop delay line onto a potential that never forgets, and it changes d by one everywhere, which changes every exploration rate and every engine's bit-agreement.

**The choices.** Zones: keep the prefix/suffix convention and say in the clause that it is an addressing convenience with no meaning to the neuron (cheap, and honest if stated); or zones as arbitrary sets, truer to the value, at the cost of a stored membership list in every checkpoint and a membership test in every wiring loop. Disjointness: disjoint by default with overlap permitted on request (current, and consistent with permissive structures); or no default, the problem stating its zones, with a refusal only where a bit has nowhere to go. Self-connections: keep the ban as a stated rule; permit them; or permit them but exclude them from the fan-in that sets the hazard.

**Blocks.** §4.4–§4.6, §2.3, the goo checkpoint format, and the fan-in clause — the self-connection answer must land before §7.2, because it changes d.

