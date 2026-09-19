# The state of the art, scored: the bibliography of Korcsak-Gorzo et al. (2025)

*Byron, September 19, 2026: "I anticipate that nearly every reference in the bibliography of the
Korcsak-Gorzo et al. preprint impinges on this project... 1 is MUST have because they directly impinge
on the system we are investigating and its scholarly ancestry... 0 is irrelevant. I will be reading the
list in order from the top down. This preprint is THE state of the art right now."*

All 96 references of arXiv:2511.21674v1, in the order they land in the bibliography, each scored from
1.0 (must read) to 0.0 (irrelevant) against walnutbutter and its scholarly ancestry. Twelve readers took
eight references each; each found the work, read at least its abstract, checked where it can be read free,
and wrote the paragraph. The preprint itself is `references/KorcsakGorzoEtAl25-arxiv.pdf`.

## The eight that scored 1.0

- **[2]** Eligibility Traces and Plasticity on Behavioral Time Scales: Experimental Support of NeoHebbian Three-Factor Learning Rules — Gerstner, Lehmann, Liakoni, Corneil & Brea
- **[4]** Neuromodulated Spike-Timing-Dependent Plasticity, and Theory of Three-factor Learning Rules — Frémaux & Gerstner
- **[6]** A solution to the learning dilemma for recurrent networks of spiking neurons — Bellec, Scherr, Subramoney, Hajek, Salaj, Legenstein & Maass
- **[46]** A Unified Framework of Online Learning Algorithms for Training Recurrent Neural Networks — Marschall, Cho & Savin
- **[60]** Cell-type-specific neuromodulation guides synaptic credit assignment in a spiking neural network — Liu, Smith, Mihalas, Shea-Brown & Sümbül
- **[81]** SuperSpike: Supervised Learning in Multilayer Spiking Neural Networks — Friedemann Zenke and Surya Ganguli
- **[83]** Solving the distal reward problem through linkage of STDP and dopamine signaling — Eugene M. Izhikevich
- **[92]** Learning by the dendritic prediction of somatic spiking — Urbanczik & Senn

## The nine that scored 0.9

- **[3]** Eligibility traces as a synaptic substrate for learning — Shouval & Kirkwood
- **[8]** Local online learning in recurrent networks with random feedback — Murray
- **[31]** Synaptic Modifications in Cultured Hippocampal Neurons: Dependence on Spike Timing, Synaptic Strength, and Postsynaptic Cell Type — Bi & Poo
- **[32]** Surrogate Gradient Learning in Spiking Neural Networks: Bringing the Power of Gradient-Based Optimization to Spiking Neural Networks — Neftci, Mostafa & Zenke
- **[40]** Brain-Inspired Learning on Neuromorphic Substrates — Zenke & Neftci
- **[57]** ETLP: Event-based three-factor local plasticity for online learning with neuromorphic hardware — Quintana, Perez-Peña, Galindo, Neftci, Chicca & Khacef
- **[59]** Smooth Exact Gradient Descent Learning in Spiking Neural Networks — Klos & Memmesheimer
- **[75]** Random synaptic feedback weights support error backpropagation for deep learning — Lillicrap, Cownden, Tweed & Akerman
- **[84]** An Imperfect Dopaminergic Error Signal Can Drive Temporal-Difference Learning — Wiebke Potjans, Markus Diesmann and Abigail Morrison

## How the scores fell

| score | count |
|---|---|
| 1.0 | 8 |
| 0.9 | 9 |
| 0.8 | 8 |
| 0.7 | 8 |
| 0.6 | 10 |
| 0.5 | 11 |
| 0.4 | 8 |
| 0.3 | 11 |
| 0.2 | 10 |
| 0.1 | 9 |
| 0.0 | 4 |

Note: **[87] and [93] are the same work** — the conference paper and its arXiv version — so the list holds
95 distinct works. Both are scored alike.

---

# The list, in bibliography order

## [1] — 0.3

**Roy, Jaiswal & Panda (2019), "Towards spike-based machine intelligence with neuromorphic computing", Nature 575(7784):607-617, doi:10.1038/s41586-019-1677-2**

This is a broad Nature review of the whole neuromorphic-computing field: spiking neurons and event-driven encoding as a route to machine intelligence that costs far less energy than conventional deep learning, surveyed from the algorithms down to the silicon. It walks through how spikes encode information, what learning rules people use in spiking networks (spike-timing-dependent plasticity, conversion from trained rate networks, surrogate-gradient training), and then spends most of its length on hardware: CMOS chips like TrueNorth and Loihi, memristive and other post-CMOS devices, and the case for designing algorithm and hardware together. It is a map of a field, not a result. For walnutbutter it touches almost nothing you have built: you are not targeting a chip, and the learning-rule material is a summary of things covered far better by references 2, 4 and 6 on this same list. The one thing it would give you is a sense of why so much of this literature is shaped by hardware constraints, which is useful for reading the e-prop preprint's motivation but changes nothing in your system. I would skip it unless you want the field's opening argument in one place, and note that the full text is behind Nature's paywall, so even the effort of getting hold of it is not repaid.

*Cited in the preprint for:* Cited once in the preprint's opening sentence as the reference for brain-inspired models offering a route to sustainable, energy-efficient AI.

*Readable:* paywalled — https://www.nature.com/articles/s41586-019-1677-2

## [2] — 1.0

**Gerstner, Lehmann, Liakoni, Corneil & Brea (2018), "Eligibility Traces and Plasticity on Behavioral Time Scales: Experimental Support of NeoHebbian Three-Factor Learning Rules", Frontiers in Neural Circuits 12:53, doi:10.3389/fncir.2018.00053**

This is the paper that shows the learning rule walnutbutter uses is not just a convenient piece of mathematics but something the brain appears to do. The argument runs like this: classical Hebbian learning says a synapse changes when the neuron before it and the neuron after it are active together, but that cannot explain learning from a reward that arrives seconds later. The three-factor answer is that the pre-and-post coincidence leaves a hidden, slowly decaying tag at the synapse (the eligibility trace), and a third signal broadcast later, typically dopamine or noradrenaline, converts that tag into an actual weight change. Gerstner and colleagues gather the experimental evidence that this really happens: in hippocampus, cortex and striatum, pairing followed by a neuromodulator up to a second or more later produces plasticity, while the neuromodulator alone or the pairing alone produces none. They then connect those timescales to reinforcement learning theory and to the behavioural delays animals actually bridge. For you this is the empirical spine of your global reward signal multiplied by a per-synapse eligibility: it tells you what a real eligibility trace decays like, and it is the reference that justifies your shaping function of spike timing being a separate thing from the reward. It is free on arXiv and in PubMed Central, so read it early.

*Cited in the preprint for:* Cited (with reference 3) as the extensive experimental evidence supporting three-factor learning models, in the sentence that introduces e-prop's class of rules.

*Readable:* open access — https://pmc.ncbi.nlm.nih.gov/articles/PMC6079224/

## [3] — 0.9

**Shouval & Kirkwood (2025), "Eligibility traces as a synaptic substrate for learning", Current Opinion in Neurobiology 91:102978, doi:10.1016/j.conb.2025.102978**

A short, recent review that updates reference 2 with seven more years of experiments, and it carries one argument that lands squarely on a design question you have open. Shouval and Kirkwood survey the experimental demonstrations of synaptic eligibility traces across several systems, and note that the trace is triggered in different places by different things: sometimes a neuromodulator, sometimes a dendritic plateau potential. The part that matters for you is their insistence that there are traces for both strengthening and weakening, and that these opposing forces give the synapse a stopping rule. Their point is that a rule with only strengthening drives every synapse to its ceiling, at which point the network has lost its selectivity and can no longer represent anything. That is exactly the job your centred Hebbian term does when it subtracts the expected firing probability from the observed spike: the subtraction is what stops the weight running away. Reading this will tell you whether biology's stopping rule is the same shape as yours or a different one, and it is a direct check on the question of what expectation the credit should be measured against. The full text is behind Elsevier's paywall and I could find no legitimate free copy; the PubMed abstract is free and is substantive enough to tell you whether it is worth chasing through a library.

*Cited in the preprint for:* Cited alongside reference 2 as experimental support for three-factor models of plasticity.

*Readable:* paywalled (free version: PubMed abstract only) — https://pubmed.ncbi.nlm.nih.gov/39965463/

## [4] — 1.0

**Frémaux & Gerstner (2016), "Neuromodulated Spike-Timing-Dependent Plasticity, and Theory of Three-factor Learning Rules", Frontiers in Neural Circuits 9:85, doi:10.3389/fncir.2015.00085**

This is already in your references folder, and it is the single closest theoretical description of what walnutbutter's learning rule is. Frémaux and Gerstner take the three-factor picture and do the mathematics: they show the conditions under which a Hebbian eligibility trace multiplied by a global reward signal actually performs gradient ascent on expected reward, which is to say when it is a genuine policy-gradient method rather than a plausible-looking heuristic. Two of their results bear directly on decisions you have made. First, the reward signal has to have its mean subtracted, otherwise the rule learns the wrong thing or nothing at all, which is the theoretical reason your dopamine term is a difference and not a raw reward. Second, they set out when the Hebbian factor has to be the exact score function of the neuron's firing probability and when a simpler correlation term will do, which is precisely the choice you have parameterised as hazard eligibility versus hebb eligibility. They also survey what has to be true of the neuromodulator's timing for this to work in a spiking network. If you only re-read one thing on this list, re-read this alongside Williams 1992, because between them they contain the derivation your rule rests on.

*Cited in the preprint for:* Cited (with reference 5) as the source for three-factor learning rules in which Hebbian activity is combined with a modulatory signal such as neuromodulation.

*Readable:* in references/ already — https://www.frontiersin.org/journals/neural-circuits/articles/10.3389/fncir.2015.00085/full

## [5] — 0.7

**Magee & Grienberger (2020), "Synaptic Plasticity Forms and Functions", Annual Review of Neuroscience 43:95-117, doi:10.1146/annurev-neuro-090919-022842**

A review from two experimentalists that sorts synaptic plasticity into four kinds and asks what each kind can and cannot learn. They start with plain Hebbian plasticity, which changes a synapse when activity on both sides coincides, and argue bluntly that it is too weak on its own: it has no way of knowing whether the change helped. Three-factor rules add neuromodulation and eligibility traces and so can be steered by an outcome. Genuinely supervised rules go further and add an explicit instructive signal that says what the answer should have been. Their fourth category is the interesting one: a form found in hippocampus, behavioural timescale synaptic plasticity, in which a long dendritic plateau in the postsynaptic cell writes a whole place field in one shot, over seconds, and does not require the Hebbian coincidence at all. The relevance to you is indirect but real. Your open question about a rate-aware teacher signal is a question about where you sit on their scale between a scalar reward and a genuine instructive signal, and this review is the clearest statement I know of what each rung buys you. It will not change your mechanism, but it will give you vocabulary for the choice. It is behind the Annual Reviews paywall with no free copy I could verify; the PubMed abstract is unusually informative and may be enough.

*Cited in the preprint for:* Cited with reference 4 for three-factor rules combining Hebbian learning with an additional modulatory signal.

*Readable:* paywalled (free version: PubMed abstract only) — https://pubmed.ncbi.nlm.nih.gov/32075520/

## [6] — 1.0

**Bellec, Scherr, Subramoney, Hajek, Salaj, Legenstein & Maass (2020), "A solution to the learning dilemma for recurrent networks of spiking neurons", Nature Communications 11:3625, doi:10.1038/s41467-020-17236-y**

This is e-prop itself, the paper the entire preprint is built on, and you cannot read the preprint without it. The dilemma in the title is that training a recurrent spiking network properly requires backpropagation through time, which demands that each synapse know things happening elsewhere in the network and at other moments in time, which no synapse can. Bellec and colleagues factor the exact gradient into two pieces: an eligibility trace that each synapse can compute on its own from its own pre- and postsynaptic history going forward in time, and a learning signal that arrives from outside. They then drop the part of the gradient that would need information from the future, and what is left is an online, local, three-factor rule that trains recurrent spiking networks well enough on speech and on memory tasks to be useful. They also present reward-based e-prop, in which the learning signal is a reward-prediction error and the whole thing becomes a policy-gradient method, which is the version closest to what you have. Two details will matter to you specifically: they handle the non-differentiable spike with a pseudo-derivative, a piecewise-linear surrogate, where you instead have a genuinely stochastic neuron whose firing probability is differentiable for free, and their eligibility trace is a forward-running filter rather than your per-synapse credit. Reading this is what lets you say precisely how your rule differs from theirs. Free at Nature Communications and on arXiv.

*Cited in the preprint for:* The preprint's central subject: e-prop is the algorithm being reimplemented event-driven in NEST, cited for its derivation, its delay assumptions, and as the foundation for reward-based variants.

*Readable:* open access — https://www.nature.com/articles/s41467-020-17236-y

## [7] — 0.6

**Werbos (1990), "Backpropagation through time: what it does and how to do it", Proceedings of the IEEE 78(10):1550-1560, doi:10.1109/5.58337**

Werbos's tutorial on backpropagation through time, the method e-prop is an approximation to. The idea is to take a recurrent network running over many time steps, unroll it into one very deep feedforward network with the same weights repeated at every step, and then run ordinary backpropagation backwards through the whole unrolled thing. Werbos presents it in terms of what he calls ordered derivatives, a bookkeeping scheme for differentiating through any chain of computations, and shows how it applies to system identification and forecasting as well as to neural networks. The paper is readable and gives worked examples rather than only formulas. For walnutbutter this is ancestry rather than machinery, and it is the ancestry of the branch you did not take: your rule never looks backwards in time and never needs a stored history, because the firing decision itself carries the credit. But every claim in the preprint about approximating a gradient is a claim about approximating this, and the words people use for what e-prop throws away, the future-facing terms, only mean something once you have seen the full backward pass they came out of. Read it if you want to understand what everyone in this literature is measuring themselves against; skip it if you are content to take that on trust. It is paywalled at IEEE, but a copy sits on a Carnegie Mellon deep-learning course page, which I checked resolves to the PDF.

*Cited in the preprint for:* Cited as the exact-gradient method that e-prop approximates online while avoiding nonlocality, time blocking, and symmetric feedback weights.

*Readable:* paywalled (free version: author-era copy on a CMU course page) — http://www.cs.cmu.edu/~bhiksha/courses/deeplearning/Fall.2016/pdfs/Werbos.backprop.pdf

## [8] — 0.9

**Murray (2019), "Local online learning in recurrent networks with random feedback", eLife 8:e43299, doi:10.7554/eLife.43299**

Murray's RFLO, developed at the same time as e-prop and arriving at the same answer from the rate-based side. The rule is exactly the shape of yours: each synapse keeps a local eligibility trace, built only from the activity arriving at it and the state of the cell it lands on, and that trace is multiplied by an error signal broadcast to the whole network through fixed random weights rather than through the transpose of the forward weights. Murray shows that this approximates the true gradient in recurrent networks, that it trains real tasks, and that the random broadcast works nearly as well as the mathematically correct feedback, which is the result that makes the whole family plausible for biology. He also argues that the resulting trace matches the form of plasticity experimentalists report. The preprint notes in its appendix that e-prop is essentially equivalent to RFLO, so this is the cleanest short statement of the idea you will find, uncluttered by spiking machinery. It is worth your time chiefly as a contrast: RFLO's third factor is a vector error signal delivered per neuron, where yours is a single global scalar, and reading Murray will make concrete what you give up and what you gain by insisting on the scalar. Free at eLife and in PubMed Central, and short.

*Cited in the preprint for:* Cited as the concurrently developed Random-Feedback Online Learning algorithm sharing e-prop's core idea on rate-based neurons, and again in the appendix survey of online recurrent training algorithms.

*Readable:* open access — https://elifesciences.org/articles/43299

## [9] — 0.0

**Abadi et al. (2016), "TensorFlow: Large-scale Machine Learning on Heterogeneous Distributed Systems", arXiv:1603.04467**

This is the original TensorFlow white paper from Google: it describes a software interface for expressing machine-learning computations as a graph of operations over tensors, and a runtime that can execute that same graph unchanged on a phone, a single GPU, or a cluster of hundreds of machines. There is no new learning theory in it at all — it is an engineering description of a dataflow framework, its device placement, and its distributed execution. The preprint cites it twice, both times as plumbing: e-prop was originally written in TensorFlow, and their Adam optimizer follows TensorFlow's particular reordering of the update equations so the numbers stay comparable with the original e-prop code. Nothing here touches how a neuron fires, how credit is assigned, or how a weight changes. For walnutbutter it is doubly irrelevant — Byron has written three engines from scratch with no framework at all, and the one substantive detail (a numerically reordered Adam) belongs to an optimizer he does not use. Skip it entirely; if the Adam reordering ever matters, reference [21] (Kingma and Ba) is the place to look, not this.

*Cited in the preprint for:* Named as the framework the original time-driven e-prop was implemented in, and as the source of the particular reordering of the Adam update equations the authors follow for comparability.

*Readable:* preprint — https://arxiv.org/abs/1603.04467

## [10] — 0.6

**Morrison, Aertsen & Diesmann (2007), "Spike-Timing-Dependent Plasticity in Balanced Random Networks", Neural Computation 19(6):1437-1467, doi:10.1162/neco.2007.19.6.1437**

This paper asks whether a large, sparsely connected network held in the irregular low-rate regime that cortex actually sits in can survive having spike-timing-dependent plasticity switched on, and finds that most published STDP rules destroy it. The authors go back to the experimental data and propose their own rule — depression scaling in proportion to the current weight, potentiation scaling as a power law of it — and show that with this rule a large balanced network stays in the asynchronous irregular regime, the weights settle into a single-humped distribution that keeps fluctuating rather than splitting into strong and weak groups, and no structure spontaneously develops. They also stimulate a group of neurons synchronously and find that the group decouples itself from the rest of the network so thoroughly it can no longer drive anyone. The preprint leans on it three separate times, and the third is the interesting one for Byron: this is where the ArchivingNode idea comes from — the postsynaptic neuron keeps a history of its own spikes, and an arriving presynaptic spike reaches back into that history to compute its own weight change, which is exactly the trick that lets the update happen at the synapse, on a spike, with no clock. That is the same architectural shape as Byron's open question about moving the exploration to the synapse, and the weight-stability half of the paper is a warning worth having: a local rule that looks fine on paper can quietly drive every weight to an extreme once you run it forever on a sparse network. Marked down from a 1.0 only because the rule itself is two-factor STDP with no reward term, which is not what walnutbutter computes.

*Cited in the preprint for:* Cited for the computational advantage of event-driven synaptic updates, as an example of hybrid event/time-driven algorithms, as a reference for STDP itself, and above all as the precedent for the ArchivingNode class the authors build EpropArchivingNode on.

*Readable:* author copy — https://brainworks.biologie.uni-freiburg.de/2007/journal%20papers/morrison-neco-2007.pdf

## [11] — 0.5

**Orchard, Jayawant, Cohen & Thakor (2015), "Converting Static Image Datasets to Spiking Neuromorphic Datasets Using Saccades", Frontiers in Neuroscience 9:437, doi:10.3389/fnins.2015.00437**

This is the paper that made N-MNIST, the dataset the preprint trains on. The problem it solves is that a spiking network wants spikes, and MNIST is a pile of still images; the usual fix is to invent an encoding in software, which the authors argue bakes in artefacts. Instead they mounted an event camera — a sensor whose pixels report only changes in brightness, each as a timestamped event — on a motorised pan-tilt head, displayed each MNIST digit on a monitor, and moved the sensor itself through three short 100 ms sweeps tracing a triangle, imitating the small involuntary eye movements humans make constantly. Each digit comes out as a 34 by 34 grid of timestamped on/off events rather than a grey-level array, with real sensor noise included; they did the same for Caltech101. For Byron this does not touch the learning rule at all, but it lands squarely on a question he has to answer anyway: how does MNIST get into a spiking network in the first place. It is worth reading as a considered answer to that question — motion generates the spikes, nothing fires where nothing changes, and a good fraction of the input is silent — and as background for reading the preprint's own MNIST results, since their task is not the same task as his. Free and short.

*Cited in the preprint for:* Cited as the source of the N-MNIST dataset used as the preprint's classification benchmark, and for the description of how the dataset was recorded.

*Readable:* open access — https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2015.00437/full

## [12] — 0.1

**Korcsak-Gorzo, Stapmanns, Espinoza Valverde, Plesser, Dahmen, Bolten, van Albada & Diesmann (2024), "Event-Based Eligibility Propagation with Additional Biologically Inspired Features", poster, NEST Conference 2024**

This is a conference poster by the same eight authors, presented by Espinoza Valverde at the NEST Conference in June 2024, and it is the early version of the preprint Byron already has. The abstract page says only what the full paper says at length: e-prop has been given extra biologically inspired features, moved into NEST's event-driven machinery, and the learning performance of the modified method is on a par with the original. No poster file was ever attached to the conference record, so the abstract on the conference site is all there is to read — a short paragraph, not a document. There is nothing in it that is not in the preprint in more detail and with the numbers attached. The only reason to open the link at all is curiosity about when this work was first shown in public. Byron should read the preprint and skip this.

*Cited in the preprint for:* Cited once, in a single sentence, to note that preliminary results were presented in abstract form.

*Readable:* proceedings (free) — https://events.hifis.net/event/1168/contributions/9973/

## [13] — 0.0

**Ansel et al. (2024), "PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation", ASPLOS '24, doi:10.1145/3620665.3640366**

The PyTorch 2 paper describes two pieces of machinery, TorchDynamo and TorchInductor, that let PyTorch keep its run-as-you-go Python style while still compiling. Dynamo hooks into the CPython interpreter at the bytecode level, watches the Python program actually execute, and extracts the tensor operations it sees into a graph, bailing back to plain Python whenever it meets something it cannot capture; Inductor then compiles that graph down to fast kernels. It is a compilers-and-systems paper, published at a computer architecture venue, and its contribution is entirely about making Python-defined models run faster on GPUs. The preprint cites it exactly once, in a throwaway clause noting that conventional artificial networks update everything in lockstep so matrix multiplications suit them, as in TensorFlow and PyTorch. That is a passing contrast, not an argument. Nothing in it bears on spiking neurons, escape noise, eligibility traces, or credit assignment, and Byron's three engines have no relationship to it. Zero.

*Cited in the preprint for:* Named in passing, alongside TensorFlow, as an example of a framework in which everything is updated synchronously at each step so matrix multiplication suits it.

*Readable:* open access — https://docs.pytorch.org/assets/pytorch2-2.pdf

## [14] — 0.7

**Buzsaki & Mizuseki (2014), "The log-dynamic brain: how skewed distributions affect network operations", Nature Reviews Neuroscience 15(4):264-278, doi:10.1038/nrn3687**

This is a review with one central claim, made over and over with different data: almost nothing in the brain is distributed in a bell curve. Firing rates of individual neurons, the strengths of synapses between pairs of cells, the number of contacts one cell makes on another, the sizes of boutons, the strengths of connections between whole brain areas — all of them come out strongly skewed with a long tail, and usually close to lognormal, spanning several orders of magnitude. A small minority of cells fire fast and carry most of the activity; the large, slow, weakly connected majority is not noise but the reserve that gives the network its flexibility and its precision. The authors argue this is not a measurement artefact but a design principle, and that it changes how you should average and analyse data in the first place. The preprint cites it for one number — human cortical rates peaking near 3 spikes per second — but the whole paper matters more to Byron than that number does. His firing hazard is exponential in the margin above threshold, which is precisely the kind of rule that turns a roughly symmetric spread of membrane potentials into a heavy-tailed spread of rates, so this is the empirical statement of what that hazard shape ought to be producing. It also gives him a concrete way to read his running network's health: if his rate histogram is not skewed, something is wrong. Free at PubMed Central and genuinely readable.

*Cited in the preprint for:* Cited for the fact that firing rates follow a lognormal distribution peaking around 3 spikes per second in human cortex, which is what makes spikes rare from a synapse's point of view and event-driven updates worth doing.

*Readable:* open access — https://pmc.ncbi.nlm.nih.gov/articles/PMC4051294/

## [15] — 0.3

**Shoham, O'Connor & Segev (2006), "How silent is the brain: is there a 'dark matter' problem in neuroscience?", Journal of Comparative Physiology A 192(8):777-784, doi:10.1007/s00359-006-0117-6**

A short review arguing that large parts of the brain are far quieter than the textbooks imply. The authors gather evidence from several recording methods that many neurons fire almost never, or only to one very specific stimulus, and point out the catch: the standard way of measuring activity in a live animal, poking an electrode in and listening, can only find a cell by hearing it spike, so the quiet ones are systematically invisible. They call these cells dark neurons by analogy with dark matter in astronomy, and close by reviewing the imaging and recording advances that might eventually settle how silent the brain really is. The preprint cites it for one cautious sentence: the true firing rates may be even lower than the lognormal estimates suggest, because of that sampling bias. For Byron the value is a framing one rather than a mechanistic one — it is evidence that a network in which most units rarely fire is a normal brain rather than a broken one, which fits how he reads his own running network. But it is a review of experimental methodology and says nothing about plasticity, credit assignment, or firing hazards, and it is paywalled, so it does not earn much of his limited time. Reference [14] makes the more useful version of the same point and is free.

*Cited in the preprint for:* Cited for the claim that measured firing rates are probably overestimates, because rarely active neurons are underrepresented in extracellular electrode recordings.

*Readable:* paywalled (free version: abstract only, at https://pubmed.ncbi.nlm.nih.gov/16550391/) — https://doi.org/10.1007/s00359-006-0117-6

## [16] — 0.4

**Morrison, Mehring, Geisel, Aertsen & Diesmann (2005), "Advancing the Boundaries of High-Connectivity Network Simulation with Distributed Computing", Neural Computation 17(8):1776-1801, doi:10.1162/0899766054026648**

This is the paper that laid down the simulation strategy NEST still runs on, and the whole preprint sits on top of it. The problem it tackles is that a cortical neuron has on the order of ten thousand incoming synapses, so the synapses, not the neurons, are what fills the machine; the answer is to split the network across many computers and, crucially, to treat neurons and synapses as different kinds of object running on different clocks — neuron states stepped on a fixed time grid, spike delivery and synaptic work done only when a spike actually happens. They show this scales to networks orders of magnitude larger than had been simulated before while still supporting varied neuron and synapse models. Two specific points from it get reused in the preprint and are worth Byron's attention. The first is that because a spike takes real time to travel down an axon, the sender does not have to be in step with the receiver: the transmission delay buys you a window in which the two time scales can be decoupled, and each process can run ahead independently until the next exchange. Byron has discrete hop delays between neurons and has not yet cashed in what they permit. The second is placing each synapse on the same machine as its postsynaptic neuron, so no message is needed to fetch postsynaptic state. It is a systems paper rather than a learning paper, it changes nothing about his rule, and unfortunately it is behind MIT Press's paywall with no free copy I could find anywhere — abstract only. Read it if the delay argument grabs him; otherwise reference [18] covers the modern version of the same story and is open access.

*Cited in the preprint for:* Cited as foundational work on hybrid event- and time-driven simulation, for the point that transmission delays let the neuronal and synaptic time scales be decoupled, and for the rule that synapses are allocated on the same MPI process as their postsynaptic neuron.

*Readable:* paywalled — https://doi.org/10.1162/0899766054026648

## [17] — 0.8

**Morrison, Diesmann & Gerstner (2008), "Phenomenological models of synaptic plasticity based on spike timing", Biological Cybernetics 98, 459-478, doi:10.1007/s00422-008-0233-1**

This is a long review paper that tries to put every practical model of synaptic plasticity into one framework: short-term depression and facilitation, pair-based spike-timing-dependent plasticity, triplet rules, voltage-dependent rules, and the whole question of how a rule that only has access to spike times, membrane voltage, the current weight, and low-pass filtered versions of those can still reproduce what experimenters measure in slices. It deliberately restricts itself to rules that are cheap enough to run in a large network simulation, which is exactly the constraint you work under. The part that matters most for you is the closing discussion, where the authors connect these local timing rules to teacher-based (supervised) and reward-based (reinforcement) learning -- that is, to the three-factor family your rule belongs to -- and they are explicit about which local quantities a synapse would have to keep around for each. Wulfram Gerstner is a co-author, so this is the same lineage as your escape-noise neuron. I scored it high because it is the cleanest single place to see what everyone else means by a "Hebbian" term, what a low-pass filtered eligibility is supposed to be doing, and why weight-dependence is a live argument -- all decisions you have made in walnutbutter without necessarily seeing the menu they were chosen from. It will not tell you anything about your global scalar reward signal directly, and it predates e-prop by a decade, so read it as the map of the territory rather than as a rival design. The preprint cites it only in passing, as one of the hybrid event-and-time-driven simulation schemes.

*Cited in the preprint for:* Cited in Section 3.1.1 as one of the body of work on hybrid event-driven-synapse / time-driven-neuron simulation algorithms that the preprint's own scheme follows.

*Readable:* open access — https://pmc.ncbi.nlm.nih.gov/articles/PMC2799003/

## [18] — 0.2

**Jordan, Ippen, Helias, Kitayama, Sato, Igarashi, Diesmann & Kunkel (2018), "Extremely Scalable Spiking Neuronal Network Simulation Code: From Laptops to Exascale Computers", Frontiers in Neuroinformatics 12, 2, doi:10.3389/fninf.2018.00002**

This paper is about the plumbing inside the NEST simulator. When you try to simulate a brain-scale network, each neuron still has only about ten thousand incoming connections, so the connectivity matrix becomes extraordinarily sparse, and the old approach -- where every compute node hears about every spike -- wastes almost all of its communication. The authors build a two-tier connection data structure and a directed communication scheme so that a node only receives spikes it actually has targets for, and they show it scales to post-petascale machines without slowing down small runs on a laptop. There is one idea here that brushes against your work: they point out that when you grow a network while holding the number of synapses per neuron fixed (rather than letting it grow quadratically), the firing rate stays roughly constant while correlations fall -- which is the same family of concern as your kappa(N) = sqrt(60/N) hazard scaling, though they reach it by fixing in-degree rather than by scaling the hazard. Otherwise this is high-performance-computing engineering for a simulator you do not use, written for people running MPI jobs on supercomputers. Read it only if you find yourself wondering how NEST gets its numbers, or if you ever want the argument for why constant in-degree is the biologically honest way to scale a network.

*Cited in the preprint for:* Cited twice: as evidence that hybrid event/time-driven algorithms suit parallel and distributed computing, and (in the scaling methods) for the fact that holding in-degree constant while growing the network keeps the firing rate approximately constant while correlations decrease.

*Readable:* open access — https://pmc.ncbi.nlm.nih.gov/articles/PMC5820465/

## [19] — 0.1

**Kurth, Senk, Terhorst, Finnerty & Diesmann (2022), "Sub-realtime simulation of a neuronal network of natural density", Neuromorphic Computing and Engineering 2(2), 021001, doi:10.1088/2634-4386/ac55fc**

The claim here is a speed record of a particular kind: the authors get a full-scale model of a cortical microcircuit -- about eighty thousand neurons at natural synapse density, with every connection explicitly represented -- to run faster than the biological time it simulates, on ordinary compute hardware rather than a supercomputer. The engineering is careful work on the NEST kernel: better use of caches and threads, tighter spike delivery, and attention to where the wall-clock actually goes. Their argument for why sub-realtime matters is not robotics alone but the study of slow processes -- learning, development -- that take hours or days of biological time and would otherwise be out of reach. For you this is almost entirely other people's furniture. Your engines are your own, your bottleneck is core count rather than MPI communication, and none of the design choices here touch your firing rule, your eligibility, or your reward signal. I scored it low deliberately. The one sentence worth carrying away is their motivation: if you want to watch a network live for a long time rather than converge, you need it to run faster than the clock -- which is the same reason you care about wall-clock in your sweeps.

*Cited in the preprint for:* Cited alongside [18] in Section 3.1.1 as demonstration that hybrid event- and time-driven simulation algorithms are suited to parallel and distributed computing.

*Readable:* preprint — https://arxiv.org/abs/2111.04398

## [20] — 0.5

**Stapmanns, Hahne, Helias, Bolten, Diesmann & Dahmen (2021), "Event-based Update of Synapses in Voltage-Based Learning Rules", Frontiers in Neuroinformatics 15, 609147, doi:10.3389/fninf.2021.609147**

This is the direct methodological parent of the preprint you are reading. The problem it solves: classic spike-timing plasticity only needs spike times, so a synapse can sleep until a spike arrives and then do all its arithmetic at once. But newer rules (Clopath's voltage-dependent rule, and the Urbanczik-Senn dendritic prediction rule) need the postsynaptic membrane voltage continuously, which seems to force you back to updating every synapse on every clock tick -- the thing that kills scaling. The authors' answer is to make the neuron keep an archive of the voltage-derived quantities its synapses will need, so that a synapse arriving late can reach back and reconstruct what it missed; they give two archiving algorithms, analyse the cost of each, and show that compressing or sub-sampling the stored voltage history buys a large speedup. This is where the preprint's EpropArchivingNode comes from, and where the whole trick of "update on the spike, using the interval since the last spike" was worked out. For you it matters less as simulator engineering than as a statement of the locality question you are already asking: what does a synapse need to know, and who has to remember it? Your open question about moving the exploration noise to the synapse is the same question from the other end -- if the exploring object is the synapse, then the synapse needs its own memory, and this paper is the careful accounting of what that costs.

*Cited in the preprint for:* Cited as the established archiving framework the preprint builds on (ClopathArchivingNode, UrbanczikArchivingNode, extended here into EpropArchivingNode), and as prior work implementing three-factor rules in NEST.

*Readable:* open access — https://arxiv.org/abs/2009.08667

## [21] — 0.3

**Kingma & Ba (2017), "Adam: A Method for Stochastic Optimization", arXiv:1412.6980, doi:10.48550/arXiv.1412.6980**

Adam is the optimizer that almost all of modern machine learning uses by default. The idea is small: instead of stepping the weight by the raw gradient times a learning rate, you keep two running exponential averages per weight -- one of the gradient and one of the squared gradient -- and step by the first divided by the square root of the second. The effect is that each weight gets its own effective step size, large where the gradient has been consistent and small where it has been noisy, and the method is insensitive to the overall scale of the gradient. The two averages are bias-corrected because they start at zero. The paper proves convergence in the convex case and shows it works well in practice. Its relevance to you is mostly that the preprint uses it, so you need it to read their equations 49 to 56, and they go to some trouble to show that Adam cannot simply be applied to a summed gradient -- the two moments have to be updated step by step, which is a genuine constraint on making the algorithm event-driven. Against your own values it is a counterexample: Adam is three extra knobs and two extra numbers of state per synapse, exactly the kind of machinery you have been keeping out of walnutbutter. Worth knowing what it does and why they needed it; not something to adopt.

*Cited in the preprint for:* Cited as the optimizer used for e-prop weight updates instead of plain gradient descent, with the preprint reproducing the original Adam update and TensorFlow's reordered form of it.

*Readable:* preprint — https://arxiv.org/abs/1412.6980

## [22] — 0.6

**Sabatini & Regehr (1999), "Timing of Synaptic Transmission", Annual Review of Physiology 61, 521-542, doi:10.1146/annurev.physiol.61.1.521**

This is an experimental physiology review about how fast, and how precisely, a real synapse passes a signal along. The authors walk through every step between a spike arriving at the presynaptic terminal and the postsynaptic response -- calcium channel opening, vesicle release, diffusion across the cleft, receptor kinetics -- and ask where the delay and the jitter come from. Their central point is that mammalian synapses at body temperature are much faster and much more precise than the room-temperature and invertebrate measurements everyone had been extrapolating from, and that this precision is what makes possible things like sound localisation from interaural time differences and, they note explicitly, Hebbian learning mechanisms that depend on concerted firing. The preprint cites it for one narrow purpose: to argue that e-prop's original assumption of instantaneous transmission to and from the output layer is biologically wrong, because real transmission takes measurable time. That is the sentence that should interest you. Your engines already give every connection a discrete hop delay rather than instantaneous transmission, so this is the empirical backing for a choice you made on other grounds -- and it is also the honest statement of scale, since synaptic delay is a fraction of a millisecond while the timing jitter is smaller still. It will not change your learning rule; it tells you your delays are not an artefact.

*Cited in the preprint for:* Cited as the empirical evidence that transmission delays exist in neurobiological processes, against e-prop's zero-delay treatment of transmissions to, within and from the output layer.

*Readable:* paywalled (free version: abstract only, on PubMed) — https://pubmed.ncbi.nlm.nih.gov/10099700/

## [23] — 0.4

**Jirsa (2004), "Connectivity and dynamics of neural information processing", Neuroinformatics 2, 183-204, doi:10.1385/NI:2:2:183**

A review that asks a single question systematically: if you change how a network is wired, what happens to what it does? The author sorts the literature along three axes -- the dynamics of the individual node (does it settle to a fixed point, oscillate, or go chaotic), the time delays incurred by signals travelling along the connections, and the statistical properties of the connectivity matrix itself (its symmetry, its translational invariance, its degree distribution). The conclusion is that these are not independent: the same local neuron model gives qualitatively different network behaviour at different anatomical scales, because the connectivity changes as you move from local circuit to whole cortex, and because the delays grow with distance. The preprint cites it for the claim that delays play a critical role in brain information processing, not merely a nuisance to be tolerated. For you the delay axis is the relevant one -- walnutbutter's hops are a delay structure, and this is the argument that the delays are doing computational work rather than just slowing things down. The connectivity-statistics material is also loosely adjacent to your sparse permissive wiring. But this is a broad theory review at the level of neural fields and large-scale dynamics, a long way above a synapse deciding how much to change, and nothing in it will reach into your learning rule.

*Cited in the preprint for:* Cited for the claim that transmission delays play a critical role in brain information processing and representation, supporting the preprint's rejection of instantaneous transmission.

*Readable:* paywalled (free version: abstract only, on PubMed) — https://link.springer.com/article/10.1385/NI:2:2:183

## [24] — 0.1

**Allan, Jones, Lee & Allan (1995), "Software pipelining", ACM Computing Surveys 27(3), 367-432, doi:10.1145/212094.212131**

This is a compiler survey, not a neuroscience paper. Software pipelining is the technique by which a compiler rearranges the instructions of a loop so that operations from several successive iterations are in flight at once -- iteration three's load happening while iteration two's multiply and iteration one's store are still executing -- so that a machine with multiple functional units is kept busy. The survey covers the two main families, modulo scheduling and kernel recognition, and compares the many published algorithms for each. The preprint borrows it purely as a metaphor: if your simulator forces at least one time step of delay on every transmission, then a chain of transmissions that the original e-prop derivation treated as instantaneous can be understood as a pipeline unrolled across several steps, with a fixed number of learning-signal values missing at the end of each sample. That is a nice way to think about it, and the accounting that follows -- three instantaneous transmissions becoming three missing learning signals -- is worth understanding when you read Section 3.1.3. But you do not need this survey to understand the metaphor, and there is nothing in its six hundred-odd references that touches walnutbutter. Skip it.

*Cited in the preprint for:* Cited in Section 3.1.3 for the idea that instantaneous transmissions can be reinterpreted as a pipeline unrolled over multiple steps once the framework imposes non-zero transmission delays.

*Readable:* open access — https://pages.cs.wisc.edu/~fischer/cs701.f14/softpipe.pdf

## [25] — 0.5

**Laje & Buonomano (2013), "Robust timing and motor patterns by taming chaos in recurrent neural networks", Nature Neuroscience 16(7):925-933, doi:10.1038/nn.3405**

A randomly wired recurrent network of rate units, driven hard enough, produces chaotic activity: the same input never gives the same trajectory twice, so nothing downstream can rely on it. Laje and Buonomano train the recurrent weights so that one particular trajectory the untrained network happened to produce becomes locally stable - they call it the network's "innate" trajectory - and show that afterwards the network reproduces that trajectory reliably under noise, can time intervals of a second or more, and can drive complex two-dimensional handwriting-like motor patterns. They also point out this matches the experimental observation that neural variability drops after a stimulus arrives. The preprint cites it only in passing, as the inspiration for training two-dimensional patterns with two output neurons in its pattern-generation task. For you this is context rather than machinery: it is a rate network, not spiking, and its learning is an offline least-squares fit to a recorded target trajectory - the opposite of your global scalar reward with no target trajectory anywhere. What earns it a middling score is the underlying question, which is one of yours: how a recurrent network that is intrinsically noisy can still be made to produce something reliable, and what "reliable" even means when the network never stops running. Worth an hour if you want the rate-network view of that problem; skip it if you only have time for the learning-rule lineage.

*Cited in the preprint for:* Cited as the literature that inspired training two-dimensional output patterns with two output neurons in the pattern-generation task.

*Readable:* open access — https://pmc.ncbi.nlm.nih.gov/articles/PMC3753043/

## [26] — 0.4

**Carandini & Heeger (1994), "Summation and Division by Neurons in Primate Visual Cortex", Science 264(5163):1333-1336, doi:10.1126/science.8191289**

This is the paper that put divisive normalization on the map. Recording from simple cells in monkey primary visual cortex, Carandini and Heeger showed that a cell first adds up its thalamic inputs linearly, and then its response is divided by the pooled activity of a large population of neighbouring cells - and that the cell membrane itself does the arithmetic, because synaptic currents sum while the total membrane conductance divides. It became the template for what is now treated as a canonical cortical computation. The preprint cites it, together with reference 27, for one narrow purpose: e-prop's classification loss needs a softmax, softmax divides each output neuron's exponential by the sum over all output neurons, and that requires every output neuron to see every other one - so the authors point at normalization in real cortex as evidence that such divisive pooling is not biologically absurd. For walnutbutter this is a side door. You have no softmax and no cross-neuron division; your reward is a single global scalar and your expectation is per-synapse. Reading it would mostly tell you what the alternative design looks like and why the preprint felt it had to defend it - which is why the preprint then goes on to drop softmax anyway in favour of squared error (reference 28). The 1994 paper is behind Science's paywall; their 2012 Nature Reviews Neuroscience review "Normalization as a canonical neural computation" covers the same ground more fully and is free.

*Cited in the preprint for:* Cited as evidence that divisive normalization mechanisms (which the softmax over output voltages requires) exist in the brain.

*Readable:* paywalled (free version: the authors' 2012 review "Normalization as a canonical neural computation", PMC3273486) — https://pmc.ncbi.nlm.nih.gov/articles/PMC3273486/

## [27] — 0.3

**Wilson, Runyan, Wang & Sur (2012), "Division and subtraction by distinct cortical inhibitory networks in vivo", Nature 488(7411):343-348, doi:10.1038/nature11347**

Using optogenetics to switch on specific classes of inhibitory neuron in mouse visual cortex while imaging the excitatory cells, Wilson and colleagues found that two kinds of interneuron do two different pieces of arithmetic. Parvalbumin cells scale responses down proportionally - a division, a gain change, which leaves each cell's orientation preference intact. Somatostatin cells subtract a roughly constant amount, which shifts the baseline down and sharpens tuning by clipping the weak responses away. The point of the paper is that the zoo of inhibitory cell types is not redundancy; the types correspond to distinct operations. The preprint cites it alongside reference 26 for the same single sentence: divisive normalization, which the softmax needs, has a known cortical implementation. For you this is the weakest link in your assignment. Nothing in walnutbutter divides by a pooled activity, you have no interneuron classes, and your design value that neurons are first-class citizens with no role given by position or label cuts directly against a story built on labelled cell types. Read it only if the question "what would it mean to give some neurons a different job" ever becomes live for you; otherwise it is a nice piece of mouse physiology that leaves your system unchanged.

*Cited in the preprint for:* Cited jointly with reference 26 as in-vivo evidence for divisive normalization mechanisms in the brain, justifying the softmax's cross-neuron communication.

*Readable:* open access — https://pmc.ncbi.nlm.nih.gov/articles/PMC3653570/

## [28] — 0.5

**Hui & Belkin (2021), "Evaluation of neural architectures trained with square loss vs cross-entropy in classification tasks", arXiv:2006.07322 (extended version of the ICLR 2021 paper), doi:10.48550/arXiv.2006.07322**

Everyone trains classifiers with cross-entropy because everyone trains classifiers with cross-entropy. Hui and Belkin went and checked. Across a large spread of architectures and benchmarks - natural language tasks, speech recognition, and vision - they retrained with plain squared error against one-hot targets, holding hyperparameters fixed, and found that on nearly all the non-vision tasks squared error matched or beat cross-entropy, sometimes by a lot, with vision showing a slight edge the other way and squared-error training being less sensitive to the random seed. Their conclusion is not that squared error is better but that the received wisdom rests on much thinner evidence than its confidence suggests. The preprint leans on exactly this to justify replacing e-prop's softmax cross-entropy with a temporal mean squared error - which is what buys it strict locality, since squared error needs no sum over other output neurons, and as a bonus it cuts the number of in-flight learning signals by one. For you the value is the argument, not the experiments: it is a clean worked case of a convention being kept because it was never tested, and the thing it licenses - dropping a loss that forces neurons to talk to each other - is a move in the same direction as your own refusal of artificial restrictions. It will not change your learning rule, since you do not differentiate a loss at all, but it is directly useful if the rate-aware teacher signal ever needs a target.

*Cited in the preprint for:* Cited to justify replacing e-prop's softmax cross-entropy loss with a temporal mean squared error, on the grounds that square loss performs comparably across architectures and benchmarks.

*Readable:* preprint — https://arxiv.org/abs/2006.07322

## [29] — 0.4

**Zucker & Regehr (2002), "Short-Term Synaptic Plasticity", Annual Review of Physiology 64(1):355-405, doi:10.1146/annurev.physiol.64.092501.114547**

This is the standard fifty-page review of the fast, reversible ways a synapse changes its own strength over milliseconds to minutes: facilitation, augmentation, post-tetanic potentiation and depression. The mechanisms are mostly presynaptic and mostly about calcium - residual calcium left in the terminal after a spike makes the next release more likely, while depletion of the ready pool of vesicles makes it less likely - and the balance between them means a synapse's gain depends on the recent history of spikes through it, not on any external signal. Zucker and Regehr lay out the evidence for each mechanism and the arguments about which calcium sensor does what. The preprint cites it, with reference 30, for a single structural point: when it makes every transmitted spike trigger a weight update based on the interval since the previous spike, that per-spike, interval-dependent character has a real biological precedent in short-term plasticity. For walnutbutter the review is background rather than mechanism - you have no short-term plasticity, your weights change only under the global reward - but two things in it are worth your attention. First, it is the canonical demonstration that a synapse can compute something from its own spike history alone, which is precisely the property your open question about generating exploration noise at the synapse would need. Second, it is a reminder of how much a real synapse does that your model deliberately does not. Read the introduction and the depression sections; the rest is pharmacology.

*Cited in the preprint for:* Cited (with reference 30) as the biological precedent for updating weights on every spike based on the interval since the previous spike.

*Readable:* author copy — https://mcb.berkeley.edu/labs/zucker/PDFs/Zucker_AnnRevPhysiol64,355.pdf

## [30] — 0.4

**Tsodyks, Uziel & Markram (2000), "Synchrony Generation in Recurrent Networks with Frequency-Dependent Synapses", The Journal of Neuroscience 20(1):RC50, doi:10.1523/JNEUROSCI.20-01-j0003.2000**

A short, sharp paper. Tsodyks and colleagues simulated a randomly wired recurrent network of excitatory and inhibitory neurons in which every synapse is frequency-dependent - its strength depends on how recently it has been used, the depression-and-facilitation model that now carries Tsodyks's name. With no external drive and no tuning, the network settles into intermittent activity that occasionally swings upward into a burst in which essentially every neuron fires within a few milliseconds, then collapses again as the synapses deplete. Synchrony, in other words, falls out of the synaptic dynamics rather than having to be imposed. The preprint cites it beside reference 29 as further precedent for spike-interval-dependent synaptic change. Its interest for you is not the citation but the phenomenon: this is a recurrent spiking network that is never trained, never converges, and just keeps living - your own reading of walnutbutter's results as health rather than convergence - and it shows how a network at the wrong operating point falls into whole-network synchrony that destroys any information the individual spikes carried. That is a failure mode worth being able to recognise in your own runs, and it bears on why a hazard that scales down as the network grows is the right instinct. Short enough to read in a sitting.

*Cited in the preprint for:* Cited (with reference 29) as short-term-plasticity precedent for the scheme that updates weights on every spike using the inter-spike interval.

*Readable:* open access — https://pmc.ncbi.nlm.nih.gov/articles/PMC6774142/

## [31] — 0.9

**Bi & Poo (1998), "Synaptic Modifications in Cultured Hippocampal Neurons: Dependence on Spike Timing, Synaptic Strength, and Postsynaptic Cell Type", Journal of Neuroscience 18(24):10464-10472, doi:10.1523/jneurosci.18-24-10464.1998**

This is the measurement that made spike-timing-dependent plasticity a fact rather than a proposal. Bi and Poo paired pre- and postsynaptic spikes at controlled delays in cultured hippocampal neurons and produced the curve everyone now draws from memory: if the presynaptic spike arrives within about twenty milliseconds before the postsynaptic one, the synapse strengthens; if it arrives after, the synapse weakens; and the effect dies away outside a window of a few tens of milliseconds. They also showed the change depends on how strong the synapse already was - weak synapses potentiate far more than strong ones, which is a built-in brake on runaway growth - and that it depends on what kind of cell sits on the postsynaptic side. The preprint cites it, with reference 10, as the empirical STDP result that licenses its per-spike weight updates. You already have this one: it is in your references folder as BiPoo98.pdf. It sits this high because it is the empirical floor under the Hebbian half of every three-factor rule, yours included - your hebb eligibility is a centred Hebbian term multiplied by a shaping function of spike timing, and this is where the claim that timing is what matters, in that direction, over that window, actually comes from. The strength dependence is the part most worth rereading against your own design, since you get that brake from a different place.

*Cited in the preprint for:* Cited (with reference 10) as the spike-timing-dependent plasticity result showing that connection strength depends on the precise timing of pre- and postsynaptic spikes, justifying per-spike weight updates.

*Readable:* in references/ already (also open access) — https://pmc.ncbi.nlm.nih.gov/articles/PMC6793365/

## [32] — 0.9

**Neftci, Mostafa & Zenke (2019), "Surrogate Gradient Learning in Spiking Neural Networks: Bringing the Power of Gradient-Based Optimization to Spiking Neural Networks", IEEE Signal Processing Magazine 36(6):51-63, doi:10.1109/MSP.2019.2931595**

The best single introduction to the central problem of training spiking networks by gradient descent, written as a tutorial rather than a results paper. The difficulty is simple to state: a spike is a step function, so its derivative with respect to the membrane potential is zero everywhere and infinite at threshold, and gradient descent has nothing to work with. The surrogate-gradient trick is to keep the hard spike in the forward pass but substitute a smooth bump - a fast sigmoid, an exponential, a piecewise-linear triangle - wherever the derivative is needed, and the paper walks through why this works, what the alternatives are, and how it relates to the older lines of attack. Crucially for you, it sets surrogate gradients alongside the stochastic-neuron approach, where the neuron fires with a probability that varies smoothly with its margin and the gradient of that probability is exact rather than invented. The preprint cites this review for the menu of surrogate functions, before choosing a smooth exponential one over e-prop's piecewise-linear one. This is a must-read for you, and not because you need a surrogate gradient - because you do not. Your escape hazard makes the firing probability a smooth function of the margin, so the score function is exact and there is no fudge; this paper is the clearest statement of what everybody else has to fudge and what it costs them, which is the argument for your design that you cannot make until you know the alternative. It also bears directly on your open question about the hazard's shape: the field's finding that performance is robust to the surrogate's exact shape is the mirror image of your question about what shape the hazard should have.

*Cited in the preprint for:* Cited for the many alternative surrogate gradient functions in the literature, which when height- and width-matched give similar shapes and comparable performance to e-prop's piecewise-linear one.

*Readable:* preprint — https://arxiv.org/abs/1901.09948

## [33] — 0.8

**Zenke & Vogels (2021), "The Remarkable Robustness of Surrogate Gradient Learning for Instilling Complex Function in Spiking Neural Networks", Neural Computation 33(4):899-925, doi:10.1162/neco_a_01367**

Spikes are all-or-nothing, so there is no derivative to descend on. The standard workaround is a surrogate gradient: pretend, for the purpose of the learning rule only, that the spike was a smooth bump-shaped function of how far the neuron's voltage sat above threshold. Zenke and Vogels ran a large systematic simulation study asking whether it matters which bump you pick. Their answer is that the SHAPE barely matters at all - box, triangle, sigmoid-derivative, exponential all train to roughly the same accuracy - but the SCALE (how wide the bump is, in voltage units) matters a great deal, and combining surrogate gradients with activity regularization lets networks work at very sparse firing rates. That separation is directly your open question (4). walnutbutter does not need a surrogate at all: because your neuron fires stochastically with hazard kappa(N) exp(s/Delta), the derivative of the log firing probability is exact, and ESCAPE_DELTA is exactly the scale parameter Zenke and Vogels found to be the one that counts. So this paper is empirical evidence from a neighbouring method that your one remaining knob is the right knob, and that fiddling with the functional form of the hazard is likely to buy you less than tuning its width. It is also the study the preprint leans on to justify swapping its piecewise-linear surrogate for an exponential one, so it is the pivot for reference 34 too.

*Cited in the preprint for:* Cited at section 3.3.7 as the evidence that surrogate gradient learning is robust to changes in the shape of the surrogate derivative, which licenses e-prop+ replacing the piecewise-linear surrogate with a smoother one.

*Readable:* open access — https://www.biorxiv.org/content/10.1101/2020.06.29.176925v1

## [34] — 0.7

**Shrestha & Orchard (2018), "SLAYER: Spike Layer Error Reassignment in Time", Advances in Neural Information Processing Systems (NeurIPS) 31**

SLAYER is a backpropagation scheme for deep spiking networks that credits error backwards across BOTH layers and time, and that learns axonal delays alongside weights - unusual, since most methods treat delays as fixed wiring. To get past the non-differentiable spike it introduces a spike-response probability density and uses an exponential function of the distance between voltage and threshold as the surrogate derivative. It shipped a GPU implementation and reported then-state-of-the-art SNN results on MNIST, N-MNIST, DVS Gesture and TIDIGITS. The preprint cites it for exactly one thing: the exponential surrogate gradient, gamma times exp(-beta times the voltage-threshold gap), which e-prop+ adopts in place of the triangle. That formula should stop you short. It is, up to a sign convention, the same exponential-in-the-margin object as your escape hazard. Where SLAYER has to POSTULATE that shape because its neurons are deterministic, your neurons are stochastic and the same shape falls out as the true derivative - the escape-noise formulation gets for free what the surrogate-gradient literature has to assume. Reading this is how you see that your hazard and their surrogate are the same curve arrived at from opposite directions. The delay-learning part is a second, softer reason to look: you have discrete hop delays that are currently fixed.

*Cited in the preprint for:* Cited as the source of the smooth exponential surrogate gradient that e-prop+ substitutes for e-prop's piecewise-linear one (Equation 28).

*Readable:* proceedings (free) — https://proceedings.neurips.cc/paper_files/paper/2018/hash/82f2b308c3b01637c607ce05f52a2fed-Abstract.html

## [35] — 0.1

**Espinoza Valverde et al. (2024), NEST, Version 3.7, Zenodo, doi:10.5281/zenodo.10834751**

This is a software release citation: the archived Zenodo record for version 3.7 of the NEST simulator, April 2024, listing its authors and changelog. There is no paper here to read, only release notes and a code archive. The preprint cites it twice - once to say that part of the e-prop implementation ships in this release, and once, more interestingly, for the "ignore-and-fire" neuron model it uses in the scaling benchmarks: a neuron that discards its synaptic input entirely and emits spikes at a fixed rate with a randomized phase, so that scaling measurements are not contaminated by the network drifting into a synchronous or runaway state. That trick is a decent benchmarking idea you could reuse if you ever want to time the Rust engine's communication cost independently of what the network is actually computing. Beyond that one paragraph there is nothing in it for walnutbutter - you are not running NEST, and the release itself has no scientific content. Skim the ignore-and-fire model page in the NEST docs if you want the idea; skip the release record.

*Cited in the preprint for:* Cited as the NEST release containing part of the e-prop implementation, and as the source of the established "ignore-and-fire" neuron mechanism used in the scaling experiments.

*Readable:* open access — https://zenodo.org/records/10834751

## [36] — 0.0

**Terhorst et al. (2025), NEST 3.9, Zenodo, doi:10.5281/zenodo.17036827**

Another software release citation: the Zenodo archive of NEST version 3.9, dated 25 September 2025. Like the 3.7 record it is a code deposit with a changelog and a long author list, not a document with an argument in it. The preprint cites it once, in a single sentence in the Discussion, to say which NEST release carries the additional e-prop+ functionality beyond what landed in 3.7 - it is a pointer to a pull request and a version number so readers can find the code. There is no science here for you at all. You are not running NEST, you have three engines of your own held to AUTHORITY.md, and nothing in a simulator's release notes bears on your hazard, your eligibility, or your dopamine rule. This is the housekeeping end of the bibliography and a clean 0.0. Listed only so you know, when you reach it in order, that you can pass straight over it.

*Cited in the preprint for:* Cited as the NEST release containing the additional e-prop+ functionality beyond release 3.7.

*Readable:* open access — https://zenodo.org/records/17036827

## [37] — 0.5

**Potjans, Morrison & Diesmann (2010), "Enabling functional neural circuit simulations with distributed computing of neuromodulated plasticity", Frontiers in Computational Neuroscience 4:141, doi:10.3389/fncom.2010.00141**

This paper solves a plumbing problem that you have already solved differently, and it is worth seeing how someone else solved it. A three-factor rule needs a global neuromodulatory signal - dopamine - to reach every synapse, but in a simulation split across a thousand machines a synapse has no idea where the dopamine-releasing neurons live. Potjans, Morrison and Diesmann introduce the "volume transmitter", an object that collects the spikes of a modulatory population and delivers that history to synapses across machine boundaries using a hybrid scheme: a bulk delivery at regular intervals plus on-demand top-ups when a presynaptic spike arrives. They show dopamine-modulated STDP scaling well to 1024 processors. For walnutbutter the interest is not the engineering - your global reward scalar is just a number every synapse can read, because you are not distributed - but the conceptual point that the third factor has a delivery LATENCY and a granularity, and that a biologically honest implementation makes the dopamine signal internally generated rather than handed down from outside. It is also the direct ancestor in this group's own work of the e-prop implementation, which is why the preprint cites it as prior art. Read it for the design pattern, not for a rule you would adopt.

*Cited in the preprint for:* Cited twice in the Discussion, as the previous effort to implement three-factor rules in NEST on which this work builds, and as the port of neuromodulated STDP that a future NESTML formalization would build on.

*Readable:* open access — https://pmc.ncbi.nlm.nih.gov/articles/PMC2996144/

## [38] — 0.0

**Gewaltig & Diesmann (2007), "NEST (NEural Simulation Tool)", Scholarpedia 2(4):1430, doi:10.4249/scholarpedia.1430**

This is the one-page Scholarpedia encyclopedia entry for NEST: what the simulator is, what it is for (large networks of point neurons, where the questions are about network dynamics rather than about the shape of a single cell), how it is built, and who maintains it. It is a tool description written for people deciding whether to use the tool. The preprint cites it in a single half-sentence to say that the work is part of a long-running collaborative project in simulation technology - it is a credit line for the simulator community, not a scientific claim the paper depends on. Nothing in it touches escape noise, eligibility, credit assignment, or anything else walnutbutter is made of. If you are ever curious what NEST actually is, this is a pleasant five-minute read and the fastest possible orientation; as a reference on your list it is irrelevant and you should pass over it.

*Cited in the preprint for:* Cited in the Discussion as the reference for NEST as a long-term collaborative project advancing neural systems simulation technology.

*Readable:* open access — http://www.scholarpedia.org/article/NEST_(NEural_Simulation_Tool)

## [39] — 0.2

**Hahne, Dahmen, Schuecker, Frommer, Bolten, Helias & Diesmann (2017), "Integration of Continuous-Time Dynamics in a Spiking Neural Network Simulator", Frontiers in Neuroinformatics 11:34, doi:10.3389/fninf.2017.00034**

Spiking simulators are built on a happy accident: neurons only need to talk to each other at spike times, so you can let every machine run independently for as long as the shortest synaptic delay and only then exchange messages. Rate-based models break that, because they need to exchange values continuously. This paper shows how to put continuous-time rate units inside NEST anyway, borrowing the iterative waveform-relaxation technique developed for gap junctions: you communicate on a coarse grid, guess the values in between, and iterate until the guesses agree. The payoff is being able to run rate models and spiking models in the same simulator and the same code, which is good for checking mean-field theory against simulation. The preprint cites it for one narrow engineering point in the Discussion - that buffering information lets you communicate on coarser time grids than the neuronal update step, which would speed things up when delays are long. For walnutbutter this is background at best. Your hop delays are discrete and your engines are single-machine; the whole problem this paper solves is one you do not have. Worth two minutes on the abstract to know the technique exists, and no more.

*Cited in the preprint for:* Cited in the Discussion as a straightforward extension that would allow transmission on coarser time grids by buffering information, promising efficiency gains for larger delays or smaller update steps.

*Readable:* open access — https://arxiv.org/abs/1610.09990

## [40] — 0.9

**Zenke & Neftci (2021), "Brain-Inspired Learning on Neuromorphic Substrates", Proceedings of the IEEE 109(5):935-950, doi:10.1109/JPROC.2020.3045625**

This is the paper that draws the map of the territory walnutbutter sits in. Zenke and Neftci start from Real-Time Recurrent Learning, the exact online alternative to backprop-through-time, which computes how every weight affects every unit's current state and is hopelessly expensive because that bookkeeping grows with the cube of the network size. They then show that if you simply DROP the off-diagonal blocks of that bookkeeping - if each synapse only tracks its effect on its own postsynaptic neuron - what falls out is a per-synapse eligibility trace multiplied by a learning signal. In other words, e-prop, SuperSpike, random-feedback online learning and the whole family of three-factor rules are all the same algorithm seen through one approximation, and the eligibility trace is not a biological guess but the surviving diagonal of an exact gradient. That is the single clearest statement of where your rule comes from and what it is an approximation to, and it is the framing you would want before deciding anything about moving the exploration noise to the synapse: this paper tells you precisely which terms locality throws away. It also surveys what neuromorphic hardware can and cannot support locally, which is the same constraint set as your own values of few knobs and permissive local structure. Of the eight in this block, read this one first.

*Cited in the preprint for:* Cited in the Discussion as the reference for RTRL-like algorithms that are practical for neuromorphic hardware, named as candidate three-factor rules to implement next in NEST.

*Readable:* preprint — https://arxiv.org/abs/2010.11931

## [41] — 0.8

**Wunderlich & Pehle (2021), "Event-based backpropagation can compute exact gradients for spiking neural networks", Scientific Reports 11:12829, doi:10.1038/s41598-021-91786-z**

This is the EventProp paper, and it is the most serious rival to the whole surrogate-gradient/e-prop family. Everyone else who wants to do gradient descent on a spiking network hits the same wall: a spike is a discontinuity, so the derivative of the output with respect to a weight does not exist in the ordinary sense, and they paper over it by substituting a smooth fake derivative. Wunderlich and Pehle instead take the continuous-time network seriously, apply the adjoint method from optimal-control theory, and work out exactly what jump each spike contributes to the backward pass. The result is a genuinely exact gradient for a continuous-time spiking network under a general loss, computed by propagating error backwards only at spike times, so the backward pass is as sparse in time and space as the forward one. Why this matters to you: walnutbutter escapes the same discontinuity by a completely different door. Because your firing is stochastic, the probability of the spike train is a smooth function of the weights even though the spikes are not, so the score function d log P / d w is exact with no surrogate and no adjoint - this paper is the sharpest available statement of the problem your escape noise dissolves, and reading it will tell you precisely what you bought by making firing a random decision. It also bears on your open question about the shape of the firing hazard: EventProp shows what the cost is when the shape is a hard threshold instead. The one thing it does not give you is locality - it is still a backward pass, not a three-factor rule.

*Cited in the preprint for:* Cited as "One example, EventProp" of RTRL-like online training algorithms that are practical for neuromorphic hardware and could be ported to NEST alongside their e-prop implementation.

*Readable:* open access — https://arxiv.org/abs/2009.08378

## [42] — 0.2

**Pehle, Blessing, Arnold, Müller & Schemmel (2023), "Event-based Backpropagation for Analog Neuromorphic Hardware", arXiv:2302.07141**

This is a progress report on getting EventProp (reference 41) to run on BrainScaleS-2, an analog neuromorphic chip built at Heidelberg. The engineering point is that the chip cannot afford to be densely measured: continuously sampling every membrane voltage would cost more energy than the computation itself. EventProp suits the chip because it only needs the spike times the system already emits, with membrane voltage measurements optional extras that can be folded in cleanly, which the authors say gives about a tenfold improvement in how much gradient information you extract per observation. They demonstrate correct gradient estimation and a small classification task on the real hardware. For walnutbutter this is essentially a hardware port with no new learning theory in it - the scientific content you would want is all in reference 41. Read it only if you ever become curious about what the analog-chip community does with these rules; nothing here changes a design decision of yours.

*Cited in the preprint for:* Cited (with 43) as evidence that EventProp has recently been implemented in an event-driven manner on the BrainScaleS neuromorphic hardware system.

*Readable:* preprint — https://arxiv.org/abs/2302.07141

## [43] — 0.1

**Billaudelle et al. (2020), "Versatile Emulation of Spiking Neural Networks on an Accelerated Neuromorphic Substrate", IEEE ISCAS 2020, pp. 1-5, doi:10.1109/ISCAS45731.2020.9180741**

This is the introductory hardware paper for BrainScaleS-2, the mixed analog-digital neuromorphic chip that reference 42 runs on. Its selling point is speed: the analog circuits run about a thousand times faster than biological real time, so an experiment that would take a simulated hour finishes in seconds, which makes long runs and many repeated trials cheap. The paper is a tour of the machine, demonstrating five different experiments to show the architecture is flexible rather than tuned to one task, with small embedded processors on the chip handling plasticity and experiment control. There is essentially no learning theory in it and nothing about eligibility traces, credit assignment or stochastic firing. For walnutbutter this is a chip datasheet dressed up as a paper. The only faintly interesting angle is that the acceleration argument is the hardware version of your own concern about wall-clock cost for a network that runs forever, but that is a stretch and not a reason to spend an hour on it.

*Cited in the preprint for:* Cited (with 42) as the BrainScaleS hardware substrate on which event-driven EventProp has been implemented.

*Readable:* preprint — https://arxiv.org/abs/1912.12980

## [44] — 0.2

**Béna, Wunderlich, Akl, Vogginger, Mayr & Gonzalez (2024), "Event-based backpropagation on the neuromorphic platform SpiNNaker2", NeurIPS 2024 Workshop on Machine Learning with new Compute Paradigms**

The same exercise as reference 42 but on a different machine: this is the first port of EventProp to SpiNNaker2, the digital many-core neuromorphic platform from Dresden. The authors discretize both the leaky integrate-and-fire equations and their adjoint equations, and ship spikes forward and error signals backward as the same kind of event packets, so the backward pass stays as sparse in communication as the forward pass - which is the whole point of doing it this way on a chip where moving data is what costs energy. They show a proof-of-concept of batch-parallel on-chip training on the Yin-Yang toy dataset, plus an off-chip version for hyperparameter searching. For walnutbutter this is again a port rather than a result: the only transferable idea is the discipline of making the learning signal travel as events on the same channel as the spikes, which is the same instinct as your global scalar dopamine broadcast. Low priority. If you want the underlying algorithm, read reference 41 instead.

*Cited in the preprint for:* Cited (with 45) as evidence that EventProp has recently been implemented in an event-driven manner on the SpiNNaker2 neuromorphic hardware system.

*Readable:* preprint — https://arxiv.org/abs/2412.15021

## [45] — 0.1

**Gonzalez et al. (2024), "SpiNNaker2: A Large-Scale Neuromorphic System for Event-Based and Asynchronous Machine Learning", arXiv:2401.04491**

This is the system paper for SpiNNaker2 itself, the chip that reference 44 runs on. It describes a digital neuromorphic processor designed so that thousands of chips can be composed into one asynchronous, event-driven machine, and surveys the range of things people intend to run on it - conventional artificial networks, spiking networks, and hybrids in between - with the motivating argument being energy cost and latency in data centres and edge devices. It is written as an architecture overview, not as an experiment. For walnutbutter there is nothing here: no plasticity rule, no credit assignment, no noise model, no result about how networks learn. Score it near zero and skip it. The one thing worth taking from it and reference 43 together is only sociological - that a large part of this field's energy goes into hardware, which is why so many of the papers in this bibliography are ports rather than ideas.

*Cited in the preprint for:* Cited (with 44) as the SpiNNaker2 hardware platform for event-driven EventProp, and again later as a substrate to which e-prop itself has been adapted.

*Readable:* preprint — https://arxiv.org/abs/2401.04491

## [46] — 1.0

**Marschall, Cho & Savin (2020), "A Unified Framework of Online Learning Algorithms for Training Recurrent Neural Networks", JMLR 21(135):1-34**

This is the map of the entire territory your learning rule lives in, and of everything in this bibliography from reference 47 to reference 60. Training a recurrent network online - updating weights as you go, without storing and replaying the past - forces a choice about how to summarize what the past did to the present, and every algorithm in the literature is a different low-rank or approximate answer to that one question. Marschall, Cho and Savin put them all in a single notation and sort them along four axes: past-facing versus future-facing (do you carry a trace of what happened, or an estimate of what a weight change will do), the tensor structure of the approximation, stochastic versus deterministic, and closed-form versus learned numerically. They then run them on matched tasks and find performance clusters by these categories rather than by how closely each one tracks the true gradient - a genuinely surprising and useful negative result. For you the payoff is two-fold: it shows that e-prop is essentially the same algorithm as Murray's RFLO, which means the state of the art you are chasing has a simpler ancestor, and it gives you the vocabulary to say exactly where walnutbutter sits - past-facing, stochastic, closed-form, with the global reward scalar doing the job their frameworks give to a learned feedback matrix. Read this one carefully and early; it is the cheapest way to learn this literature without reading thirty papers.

*Cited in the preprint for:* Cited in Section G as "A recent framework" that organizes state-of-the-art online training algorithms along criteria such as past- vs future-facing, tensor structure, stochastic vs deterministic, and closed-form vs numerical - and which shows e-prop is essentially equivalent to RFLO.

*Readable:* open access — https://www.jmlr.org/papers/volume21/19-562/19-562.pdf

## [47] — 0.7

**Tallec & Ollivier (2017), "Unbiased Online Recurrent Optimization", arXiv:1702.05043**

UORO is the honest-noise member of the online-learning family. The exact online method, real-time recurrent learning, keeps a huge array tracking how every weight has influenced every unit, which nobody can afford; the usual dodge is to truncate the history, which is cheap but biased, and Tallec and Ollivier show it can actually diverge on tasks where a weight has opposite short-term and long-term effects. Their fix is to keep a random rank-one sketch of that array instead of truncating it - each step throws in fresh random signs, the estimate is noisy but unbiased in expectation, and standard stochastic gradient convergence arguments then apply. It costs about the same as truncated backprop and converges where truncation does not. Why this should interest you: it is the clearest statement in this literature of the trade you are already making, which is to accept noise in the gradient estimate in exchange for never having to store or replay the past. Your REINFORCE-style rule is likewise unbiased and noisy, and UORO's analysis of why unbiasedness is worth paying variance for is the argument that justifies your design against the deterministic alternatives. It is a machine-learning paper with no spikes and no biology, so the connection is by analogy rather than by mechanism, which is why it is not a 1.0.

*Cited in the preprint for:* Listed in Section G as Unbiased Online Recurrent Optimization, one of the algorithms the unified framework of reference 46 compares against BPTT and RTRL.

*Readable:* preprint — https://arxiv.org/abs/1702.05043

## [48] — 0.8

**Roth, Kanitscheider & Fiete (2018), "Kernel RNN Learning (KeRNL)", International Conference on Learning Representations (ICLR)**

KeRNL is an eligibility-trace rule derived from perturbation, which puts it uncomfortably close to the centre of what you are building. Like UORO it attacks the enormous sensitivity array that exact online learning would require, but instead of a random sketch it assumes the array factorizes into two much smaller things: a sensitivity weight saying how strongly unit j's past activity matters to unit i, and a temporal eligibility kernel saying how fast that influence decays. The trick that makes it more than curve-fitting is that both of those are themselves learned, by deliberately perturbing the network and watching what happens - so the eligibility timescales are measured from the network's own behaviour rather than set by hand. On long-dependency tasks it holds its own against full backpropagation through time while needing no symmetric backward weights, no unrolled history in memory, and a much shorter feedback delay. For walnutbutter this touches two live questions at once: it is a worked example of node perturbation being converted into an eligibility trace, which is exactly the lineage of your open question about moving the exploration noise from the neuron to the synapse, and it raises a possibility you have not considered - that the eligibility time constants need not be constants you choose, but quantities the network could estimate for itself. Ila Fiete's earlier paper with Seung is already in your references folder; this is the same intellectual line twenty years on, in machine-learning clothes.

*Cited in the preprint for:* Listed in Section G as Kernel RNN Learning, one of the algorithms the unified framework of reference 46 compares against BPTT and RTRL.

*Readable:* proceedings (free) — https://openreview.net/forum?id=ryGfnoC5KQ

## [49] — 0.5

**Mujika, Meier & Steger (2018), "Approximating Real-Time Recurrent Learning with Random Kronecker Factors", NeurIPS 31**

Real-Time Recurrent Learning (RTRL) is the honest online alternative to backpropagation-through-time: instead of unrolling the past, every synapse carries a running record of how much it has influenced the current state of every neuron. That record is the exact, full-blown ancestor of what everyone now calls an eligibility trace, and it is unusable at scale because it is a matrix the size of (neurons x synapses). This paper's trick is to keep an approximation of that record as a Kronecker product of two much smaller pieces, drawn with random factors so that the approximation is unbiased and its noise stays bounded over time rather than growing; they show it learns long-range dependencies on string-memorisation and Penn Treebank about as well as truncated backprop. For Byron this is not machinery he would ever implement -- it is dense linear algebra with no locality and no biology -- but it is the clearest statement of what an eligibility trace actually approximates. Byron's per-synapse eligibility is, in this language, the crudest possible restriction of that influence record to the diagonal: the synapse only tracks its own postsynaptic neuron. Reading this tells him what he is throwing away by doing that, and how much of the field's effort goes into buying some of it back. Worth one careful pass for the framing, not for anything he would change.

*Cited in the preprint for:* Listed in Appendix G among the online RNN training algorithms that the Marschall et al. framework compares against BPTT and RTRL, as "Kronecker-Factored RTRL (KF-RTRL)".

*Readable:* proceedings (free) — https://proceedings.neurips.cc/paper/2018/hash/dba132f6ab6a3e3d17a8d59e82105f4c-Abstract.html

## [50] — 0.4

**Benzing, Gauy, Mujika, Martinsson & Steger (2019), "Optimal Kronecker-Sum Approximation of Real Time Recurrent Learning", ICML (PMLR 97), 604-613**

This is the direct sequel to reference 49 by an overlapping group, and its contribution is a proof rather than a new idea. They define the class of "Kronecker-sum" approximations to RTRL's influence record -- which turns out to contain every previously published method of this kind, including KF-RTRL and the earlier UORO -- and then show that their algorithm, called OK, is the best possible member of that class, with noise small enough to be empirically negligible. It matches truncated backprop on character-level Penn Treebank and beats it on synthetic memorisation because it can update weights online. For Byron the useful sentence is the negative one: this paper marks the ceiling of an entire approach, so if he ever wonders whether the field has a cheap exact answer to online credit assignment hiding somewhere, the answer is that the cheap answers are provably approximations and this is the best of them. Beyond that it touches nothing in walnutbutter -- no spikes, no locality, no reward signal, and the mathematics is heavier than the payoff for him. Read 49 first if he reads either; this one is only worth opening if he wants the optimality result itself.

*Cited in the preprint for:* Listed in Appendix G as "r-Optimal Kronecker-Sum Approximation (r-OK)", one of the algorithms the Marschall et al. framework compares.

*Readable:* proceedings (free) — https://proceedings.mlr.press/v97/benzing19a.html

## [51] — 0.3

**Jaderberg, Czarnecki, Osindero, Vinyals, Graves, Silver & Kavukcuoglu (2017), "Decoupled Neural Interfaces using Synthetic Gradients", ICML (PMLR 70), 1627-1635**

The problem this DeepMind paper attacks is that in ordinary backpropagation every layer must sit idle until the forward pass finishes and the error has been carried all the way back to it -- a lockstep that is both slow and biologically absurd. Their fix is to train a small auxiliary network at each interface whose job is to guess what the incoming gradient will be, from local information only; a layer then updates immediately against this guessed or "synthetic" gradient, and the guesser is itself corrected later when the true signal eventually arrives. They show feedforward layers training fully asynchronously, and recurrent networks getting an effectively longer memory horizon because the synthetic gradient stands in for the unrolled future. For Byron there is one idea here worth having in the back of his mind -- a globally broadcast learning signal can be replaced by a locally predicted one -- which is the same shape as his open question about where the exploration and the credit actually live. But the mechanism is a second trained neural network per interface, which is exactly the kind of extra apparatus his "few knobs" value rejects, and nothing about the paper is spiking, stochastic or reward-driven. Skim the introduction and the figure; the rest will not repay the time.

*Cited in the preprint for:* Listed in Appendix G as "Decoupled Neural Interfaces (DNI)", among the algorithms the Marschall et al. framework compares with BPTT and RTRL.

*Readable:* proceedings (free) — https://proceedings.mlr.press/v70/jaderberg17a.html

## [52] — 0.6

**Lee, Delbruck & Pfeiffer (2016), "Training Deep Spiking Neural Networks Using Backpropagation", Frontiers in Neuroscience 10:508**

This is one of the founding papers of the line that trains spiking networks directly rather than converting a trained rate network into spikes afterwards. The obstacle is that a spike is a discontinuity, so there is no derivative to descend; their answer is to treat the membrane potential as the differentiable signal and to carry the error through that, with the spike itself handled as a hard event sitting on top of a smooth variable. They add practical machinery that matters more than it sounds -- regularisation on the thresholds to revive neurons that have gone silent and stopped contributing -- and report 98.77 percent on permutation-invariant MNIST and 98.66 percent on the event-based N-MNIST, roughly a threefold cut in error over previous spiking networks, at about a fifth of the arithmetic of an equivalent conventional network on event data. Byron should read this for two concrete reasons: it is the same task he runs, and the dead-neuron problem it solves by threshold regularisation is the same failure mode his network faces when a unit's margin drifts so far below threshold that its escape hazard effectively stops firing it. It is not his learning rule -- it is backprop through a network, not a global reward with per-synapse eligibility -- so it will not change his design, but it is the paper that made "train the spikes directly" respectable, and the ancestor of the surrogate-gradient work the preprint leans on.

*Cited in the preprint for:* Cited in Appendix G as "a spiking backpropagation variant", one of the online training algorithms falling outside the Marschall et al. framework.

*Readable:* open access — https://www.frontiersin.org/articles/10.3389/fnins.2016.00508/full

## [53] — 0.5

**Menick, Elsen, Evci, Osindero, Simonyan & Graves (2020), "A Practical Sparse Approximation for Real Time Recurrent Learning", arXiv:2006.07232**

Another attack on RTRL's unaffordable influence record, but with a plainer idea than the Kronecker papers: keep only the entries that could have been reached within n time steps, and throw the rest away. They call it the Sparse n-step Approximation, SnAP. With n set to one -- each synapse tracks only its own immediate postsynaptic neuron -- the cost drops to that of ordinary backpropagation while beating comparable RTRL approximations; with n set to two it stays affordable specifically when the network's wiring is already sparse, and in that regime it can learn faster than truncated backprop because it is free to update the weights online. Raising n walks continuously back toward exact RTRL. This is the most directly applicable of the RTRL-approximation cluster for Byron, because SnAP-1 is essentially the eligibility he already has, and the paper states in so many words what he would gain by letting a synapse also track its postsynaptic neuron's own targets one hop further out -- which is a real design question for him given his permissive sparse wiring and his continuous, never-batched updates. The preprint also cites it for the trade-off itself: updating more often adapts faster but computes the gradient with weights that have already moved on. Read the abstract and the SnAP-1 versus SnAP-2 comparison; the rest is machine-learning engineering.

*Cited in the preprint for:* Cited twice: in Appendix G as "Sparse n-step Approximation (SnAP)", and earlier for the trade-off in truncated algorithms between dynamic adaptation and gradient accuracy when weights are updated more often than once per mini-batch.

*Readable:* preprint — https://arxiv.org/abs/2006.07232

## [54] — 0.7

**Kaiser, Mostafa & Neftci (2020), "Synaptic Plasticity Dynamics for Deep Continuous Local Learning (DECOLLE)", Frontiers in Neuroscience 14:424**

DECOLLE is a spiking network that learns online and strictly locally: instead of one error computed at the output and carried backwards, each layer is given its own cheap cost function based on a random fixed readout, so the error signal a synapse needs is available where the synapse is, at the moment it is needed, with no extra memory held for gradient bookkeeping. The plasticity rules are not invented by hand -- they are derived mechanically from the chosen cost and the neuron dynamics using ordinary automatic differentiation -- and the result is a rule of exactly the shape Byron works in: something local at the synapse multiplied by a signal arriving from elsewhere. They train on event-based data, N-MNIST and DvsGesture, and land at the state of the art of the time. This is one of the two most worthwhile items in this block for him. It is the cleanest worked demonstration that you can drop the global backward pass entirely and still train a deep spiking network, and its "local error, local eligibility, no stored history" architecture is the closest published cousin to what walnutbutter does with a global reward scalar. The honest difference is that DECOLLE's third factor is a per-layer supervised error, not a scalar reward, so it sidesteps the credit-assignment problem Byron has chosen to keep; reading it will sharpen his sense of what that choice costs him.

*Cited in the preprint for:* Listed in Appendix G as "Deep Continuous Local Learning (DECOLLE)", among the online training algorithms outside the Marschall et al. framework.

*Readable:* open access — https://www.frontiersin.org/articles/10.3389/fnins.2020.00424/full

## [55] — 0.6

**Bohnstingl, Wozniak, Pantazi & Eleftheriou (2022), "Online Spatio-Temporal Learning in Deep Neural Networks", IEEE Trans. Neural Networks and Learning Systems 34(11), 8894-8908**

The claim of OSTL is a clean factorisation: split the gradient a recurrent network needs into a spatial part, which is about how the error flows across neurons at one instant, and a temporal part, which is about how a neuron's own past feeds its present. Keep the temporal part as a running quantity local to each unit -- that is the eligibility trace -- and combine it with the spatial part as it arrives. For shallow networks they prove this gives gradients identical to backpropagation-through-time while running strictly online, and they show the same decomposition applies not just to spiking neurons but to LSTMs and GRUs, with results on language modelling and speech comparable to the BPTT baselines. For Byron the value is that this is the most explicit published account of what an eligibility trace IS as a mathematical object, and of exactly when carrying one online is equivalent to the offline gradient rather than an approximation of it. His own eligibility -- the exact score function of the escape hazard -- is a different animal, derived from a probability rather than from a derivative through time, so the equivalence result does not transfer directly. But the spatial-versus-temporal split is the right vocabulary for his open question about whether the exploring object and the credited object are the same object, and it is worth having that language. The published version is behind IEEE's paywall; the free preprint is complete and carries an extra author.

*Cited in the preprint for:* Listed in Appendix G as "Online Spatio-Temporal Learning (OSTL)", among the online training algorithms outside the Marschall et al. framework.

*Readable:* paywalled (free version: arXiv preprint 2007.12723) — https://arxiv.org/abs/2007.12723

## [56] — 0.7

**Lee, Haghighatshoar & Karbasi (2022), "Exact Gradient Computation for Spiking Neural Networks", OPT Workshop, NeurIPS**

Everyone in this literature starts from the premise that a spiking network has no usable gradient, because the spike is a hard threshold crossing and the output jumps. This paper says that premise is wrong. Applying the implicit function theorem at the spike times themselves, they prove that a spiking network does have well-defined derivatives of its output with respect to its weights -- the trick being that you differentiate not the spike, but the time at which the threshold is crossed, which moves smoothly as a weight changes. They then give a forward-propagation algorithm that computes those exact gradients by exploiting the causal ordering of spikes, and, more interestingly for Byron, they argue their result explains why Hebbian learning and the surrogate-gradient methods that everyone actually uses work as well as they do. This is the other item in this block worth real attention. Byron's escape-noise neuron reaches the same destination by a completely different road: because his firing is stochastic, the probability of the spike is already a smooth function of the weights, so his hazard eligibility is an exact score function with no implicit function theorem required. Reading this shows him what the deterministic camp has to pay to get what his noise gives him for free, and it bears directly on his open question about what shape the firing hazard should have and what that shape costs the learning rule. The workshop version cited here was later expanded into a full AISTATS 2023 paper.

*Cited in the preprint for:* Listed in Appendix G as "Forward Propagation (FP)", among the online training algorithms outside the Marschall et al. framework.

*Readable:* open access — https://arxiv.org/abs/2210.15415

## [57] — 0.9

**Quintana, Perez-Peña, Galindo, Neftci, Chicca & Khacef (2024), "ETLP: Event-based three-factor local plasticity for online learning with neuromorphic hardware", Neuromorphic Computing and Engineering 4(3), 034006, doi:10.1088/2634-4386/ad6733**

ETLP is a three-factor learning rule built to be as local as a rule can be and still learn. Each synapse keeps a running trace made of the presynaptic spike train, a smoothed measure of how close the postsynaptic neuron was to firing, and a threshold-adaptation trace; that trace is then multiplied by a third factor that arrives as spikes from dedicated teaching neurons firing at a fixed rate, one set per class, with no error ever computed anywhere. The whole thing is event-driven: a weight only moves when a spike arrives, so there is no clock. They train on event-camera MNIST and spoken digits, land a few points below e-prop (94.3 vs 97.9 percent on N-MNIST) and above DECOLLE, and build it on an FPGA to show the resource cost. For Byron this is close to home in two ways: it is a working three-factor rule where credit is a per-synapse eligibility times a global-ish broadcast signal, exactly walnutbutter's shape, and its teaching signal is delivered as a spike rate rather than as a number, which is one concrete answer to the rate-aware teacher question he has open. It also shows the price of strict locality honestly rather than hiding it, which is the trade he keeps making. The reason it is 0.9 and not 1.0 is that the paper's centre of gravity is the hardware demonstration, and the rule itself is a variant rather than a founding idea.

*Cited in the preprint for:* Listed in the appendix survey of online training algorithms for biologically plausible recurrent networks, as one of the rules outside the main framework the authors catalogue.

*Readable:* open access — https://iopscience.iop.org/article/10.1088/2634-4386/ad6733

## [58] — 0.6

**Wei, Zhang, Zhang, Belatreche, Wu, Xu, Qiu, Chen, Yang & Li (2024), "Event-Driven Learning for Spiking Neural Networks", arXiv:2403.00270, doi:10.48550/arXiv.2403.00270**

This is a survey-plus-proposal paper about making the learning step itself event-driven rather than clocked. The authors argue that most spiking-network training still walks a fixed time grid and pays for every empty time step, then offer two rules that only compute at spikes: STD-ED, which derives the weight change from the exact timing of the postsynaptic spike, and MP-ED, which derives it from the membrane potential at that moment. Both are backpropagation-style methods rather than local three-factor rules, and the evaluation is image classification accuracy (CIFAR-100 and neuromorphic datasets) plus a claimed thirty-fold energy saving on neuromorphic hardware against time-step surrogate-gradient training. For Byron the interesting part is not the accuracy table but the framing: it is the clearest recent statement of why updating on spikes and on the interval since the last spike is the right unit of work, which is the same move the e-prop preprint makes and the same one walnutbutter already makes by construction. It sits at 0.6 because the mechanism itself is not in his family -- there is no eligibility trace, no global reward, no locality -- so it will not change a design decision, but it gives him the vocabulary and the comparison points that the event-driven literature uses.

*Cited in the preprint for:* Listed in the appendix survey as the source of the STD-ED and MP-ED event-driven algorithms, among online training algorithms outside the framework of ref 46.

*Readable:* preprint — https://arxiv.org/abs/2403.00270

## [59] — 0.9

**Klos & Memmesheimer (2025), "Smooth Exact Gradient Descent Learning in Spiking Neural Networks", Physical Review Letters 134(2), 027301, doi:10.1103/physrevlett.134.027301**

This paper attacks the problem that makes gradient learning in spiking networks awkward in the first place: nudge a weight slightly and a spike can pop into existence or vanish, so the loss jumps rather than slides, and everything downstream of that spike changes at once. Klos and Memmesheimer construct neuron models in which spikes can only appear or disappear at the very end of a trial, where nothing depends on them any more, so the network's behaviour varies continuously with the weights and an exact gradient exists -- no surrogate, no smoothing, no noise. They then show this works on recurrent networks and on deep networks that start completely silent, including gradient-driven addition and removal of spikes. This matters to Byron precisely because it is the opposite bargain from the one walnutbutter strikes. He buys differentiability with escape noise: the neuron fires stochastically, the firing probability is smooth in the margin, and the score function is exact because the randomness is real. These authors buy it by engineering the deterministic dynamics instead. Reading the two side by side is the cleanest way to see what his hazard shape is actually paying for and what it would cost him to change it, which is open question four on his list. The PRL version is behind a paywall; the arXiv version is the same work.

*Cited in the preprint for:* Listed in the appendix survey as an online training algorithm that addresses the instability from spike appearance and disappearance discussed in the preprint's Subsection 3.2.

*Readable:* paywalled (free version: arXiv preprint, arXiv:2309.14523) — https://arxiv.org/abs/2309.14523

## [60] — 1.0

**Liu, Smith, Mihalas, Shea-Brown & Sümbül (2021), "Cell-type-specific neuromodulation guides synaptic credit assignment in a spiking neural network", PNAS 118(51), e2111821118, doi:10.1073/pnas.2111821118**

The question this paper asks is the one sitting underneath walnutbutter's whole learning rule: a synapse holds a local eligibility trace, and something has to tell it whether what it did was good, so what exactly is that something and how much does it need to know? The usual answers are a single global broadcast scalar, which is biologically easy but throws away nearly all the information, and full backpropagation, which is informative but requires a synapse to know things it cannot know. Liu and colleagues propose a middle answer they call MDGL, multidigraph learning: neurons release short-range neuromodulators that are specific to their cell type, so a synapse receives a learning signal that is local in space but carries which kind of neuron contributed, not just how well the whole network did. Formally the signal a cell receives is its neighbours' eligibility traces run through fixed, causal, cell-type-specific filters, so it stays local and still reaches back seconds in time. They train recurrent spiking networks obeying Dale's law and sparse wiring on delayed-decision and sequence tasks, and show the gradients MDGL produces are closer to the true ones than e-prop's are. Byron currently uses the global-scalar end of this spectrum, so this is the paper that tells him what that choice is costing him and what the cheapest upgrade would look like -- and it does so without breaking his value that no neuron gets a role from position or label, since the extra information rides on cell type, not on address.

*Cited in the preprint for:* Cited in the discussion and in the appendix survey as a rule that enhances biological plausibility by incorporating neuron-type diversity and type-specific local neuromodulation.

*Readable:* open access — https://pmc.ncbi.nlm.nih.gov/articles/PMC8713766/

## [61] — 0.1

**Plotnikov, Rumpe, Blundell, Ippen, Eppler & Morrison (2016), "NESTML: a modeling language for spiking neurons", arXiv:1606.02882, doi:10.48550/arXiv.1606.02882**

NESTML is a small domain-specific programming language for writing down neuron models, which then compiles automatically to the C++ that the NEST simulator actually runs. The argument of this 2016 paper is that general-purpose, simulator-independent model description languages end up with a lowest-common-denominator feature set, so it is worth building one that commits to a single simulator and in exchange lets you say things in the vocabulary a neuroscientist already uses -- state variables, differential equations, spike handlers -- rather than translating them into generic code by hand. There is no neuroscience result here and no learning rule; it is a tooling paper about how models get written and compiled. For Byron the only faint resonance is structural: NESTML is a specification file that generates the implementation, which is the same relationship AUTHORITY.md has to his three engines, except automated. Nothing in it touches his mechanism, his noise, or his credit assignment, and the 2025 successor below supersedes it anyway, so this is near the bottom of the range.

*Cited in the preprint for:* Cited in the discussion as the domain-specific language in which future plasticity rules might be formalized in order to port this class of online learning algorithms to NEST.

*Readable:* preprint — https://arxiv.org/abs/1606.02882

## [62] — 0.2

**Linssen, Babu, Eppler, Koll, Rumpe & Morrison (2025), "NESTML: a generic modeling language and code generation tool for the simulation of spiking neural networks with advanced plasticity rules", Frontiers in Neuroinformatics 19, doi:10.3389/fninf.2025.1544143**

This is the grown-up version of the previous entry, nine years on: the NESTML toolchain rewritten in Python, no longer tied to NEST alone (it can target SpiNNaker, with GPU work under way), and -- the substantive addition -- able to describe synapse models, not just neuron models. Neuron and synapse code are generated together, with the compiler moving variables between the two so that the result runs fast, and the paper makes a point of supporting plasticity rules where spike-timing-dependent plasticity is modulated by a third factor such as a global dopamine concentration. That last sentence is the only part of the paper that touches walnutbutter, and it touches it by coincidence rather than by contributing anything: it is the same rule shape Byron already implements by hand in three engines. What he might take from it is one engineering idea -- that a synapse model and a neuron model can be compiled jointly, with the compiler deciding which quantity lives on which side -- which is a real question in his own code whenever he considers moving the exploration noise from the neuron to the synapse. Otherwise this is infrastructure for people who run NEST, and scored accordingly.

*Cited in the preprint for:* Cited alongside ref 37 as the existing port of neuromodulated STDP, the reward-based three-factor algorithm that a NESTML-based approach would build on.

*Readable:* open access — https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2025.1544143/full

## [63] — 0.6

**Baronig, Bahariasl, Özdenizci & Legenstein (2025), "A Scalable Hybrid Training Approach for Recurrent Spiking Neural Networks", arXiv:2506.14464**

Online forward-gradient rules like e-prop have two well-known problems: they run slowly on ordinary hardware because every time step waits for the one before it, and they learn worse than backpropagation through time because the gradient they compute is only an approximation. This paper offers HYPR, which chops a long sequence into chunks, computes the forward-gradient updates for all chunks in parallel, and stitches them together -- so memory stays constant no matter how long the sequence, learning stays online in the sense that matters, and the throughput is closer to what a GPU can do. The striking result is that they nearly close the accuracy gap to backpropagation through time, and that neurons with oscillating subthreshold dynamics train unusually well under this scheme. For Byron the relevance is indirect but real: his rule is also an online, forward, approximate one, and the standing worry about that whole family is how much accuracy the approximation costs. This is the most recent honest measurement of that gap and an argument that it is smaller than people assumed. The chunked parallelism itself is aimed at GPU training rather than at a continuously living network, so it is not something he would adopt, which keeps it in the middle of the range rather than higher.

*Cited in the preprint for:* Cited in the discussion as recent work on parallelized gradient computation that reduces runtime and could guide fair, fully optimized comparisons across frameworks.

*Readable:* preprint — https://arxiv.org/abs/2506.14464

## [64] — 0.2

**Knight & Nowotny (2022), "Efficient GPU training of LSNNs using eProp", Proceedings of the Annual Neuro-Inspired Computational Elements Conference (NICE), ACM, pp. 8-10, doi:10.1145/3517343.3517346**

A three-page conference contribution reporting that the authors extended their GeNN GPU simulator so that spiking networks can be trained with e-prop on ordinary graphics hardware, borrowing tricks from machine-learning libraries such as batching many trials in parallel to keep the GPU busy. Their headline numbers are that e-prop-trained spiking classifiers match the accuracy of the same networks trained with backpropagation through time, and that inference latency and energy come out up to seven times lower than an LSTM on the same card. The one scientifically useful line for Byron is the first of those: another independent group finding that the online, local approximation does not lose to the exact method on their tasks. Everything else is GPU occupancy and kernel engineering for a simulator he does not use, and at three pages there is not much room for anything more. It is here because the e-prop preprint is surveying who has ported e-prop where, and he can read the claim in one minute; that is what a 0.2 buys.

*Cited in the preprint for:* Cited in the discussion as one of the ports of e-prop to other frameworks (mlGeNN on GPUs), and as the one such study that explores sequential MNIST and DVS gestures.

*Readable:* open access — https://sussex.figshare.com/articles/conference_contribution/Efficient_GPU_training_of_LSNNs_using_eProp/23492933

## [65] — 0.2

**James C Knight and Thomas Nowotny (2023), "Easy and efficient spike-based Machine Learning with mlGeNN", Proc. Annual Neuro-Inspired Computational Elements Conference (NICE), ACM, pp. 115-120. doi: 10.1145/3584954.3585001**

mlGeNN is a Keras-style front end that sits on top of GeNN, a GPU spiking-network simulator from Sussex. The paper's claim is convenience: you can now write down a spiking network in a few lines instead of hand-coding it, and the authors use that convenience to sweep one- and two-layer recurrent spiking networks trained with e-prop on the DVS hand-gesture dataset, ending up with a broad architectural comparison they say they could not have assembled before. There is no new learning rule here and no new theory - e-prop is used as a supplied ingredient, not examined. For walnutbutter this is a tool citation: you already have three engines of your own and a specification file that governs them, so a fourth framework with a friendlier API does not touch any decision you have open. The one thing that might interest you is the architecture sweep itself - it is a rare side-by-side of how much e-prop's accuracy depends on layer count and recurrence rather than on the rule - but that is a footnote, not a reason to read. The preprint cites it only in its closing survey of "where else has e-prop been ported", alongside its companion pyGeNN paper. Score is low because nothing in it would change your hazard, your eligibility, or your teacher.

*Cited in the preprint for:* Listed in the closing related-work paragraph as one of the frameworks e-prop has been ported to, a GPU library for sparse spike data.

*Readable:* open access — https://ndownloader.figshare.com/files/41472195

## [66] — 0.1

**James C Knight, Anton Komissarov and Thomas Nowotny (2021), "PyGeNN: A Python Library for GPU-Enhanced Neural Networks", Frontiers in Neuroinformatics 15, p. 659005. doi: 10.3389/fninf.2021.659005**

This is the layer underneath reference 65: PyGeNN exposes the C++ GeNN code generator to Python so that models can be written in Python without losing GPU speed. The substance of the paper is engineering - they show that the cost of recording spikes can dominate a simulation's runtime, introduce a new spike-recording scheme that cuts that cost by up to ten times, and then demonstrate that a full cortical column runs faster on one modern GPU than on real-time neuromorphic hardware. The single sentence in it that brushes against your work is the last one of the abstract: a smaller network with a custom three-factor learning rule, defined by the user in PyGeNN, runs nearly a hundred times faster than real time. That is a claim about code generation, not about the rule. Honestly, this is a simulator manual with benchmarks, and it is cited in the preprint as nothing more than the substrate mlGeNN stands on. Nothing in it speaks to escape noise, credit assignment, or what your eligibility should be. Read it only if you ever consider putting walnutbutter on a GPU and want to see how someone else handled the code-generation-and-sparse-data problem.

*Cited in the preprint for:* Cited once as the simulator underlying mlGeNN, in the list of frameworks to which e-prop has been ported.

*Readable:* open access — https://pmc.ncbi.nlm.nih.gov/articles/PMC8100330/

## [67] — 0.3

**Adam Perrett, Sara Summerton, Andrew Gait and Oliver Rhodes (2022), "Online learning in SNNs with e-prop and Neuromorphic Hardware", Proc. Annual Neuro-Inspired Computational Elements Conference (NICE), ACM, pp. 32-39. doi: 10.1145/3517343.3517352**

This is the Manchester group putting e-prop onto SpiNNaker 1, the first-generation ARM-based neuromorphic machine, so that a network can learn on the chip in real time rather than being trained elsewhere and downloaded. They split the work across three cores - one for input neurons, one for the hidden layer of leaky and adaptive leaky integrate-and-fire neurons, one for the readout - and train on a waveform-matching task and a temporal credit-assignment task, without recurrent connections in the hidden layer. The reason it stands slightly above the other hardware ports in this block is the thing the preprint singles it out for: like reference 69, it applies weight updates event-driven, on spikes, rather than on a clock tick. That is exactly the move the Korcsak-Gorzo preprint makes its central contribution, and it is also structurally the same as what walnutbutter already does - your network has no update clock, it has hops and spikes. Seeing someone else forced into that design by hardware, years earlier, is a useful sanity check that spike-triggered updates are a real architecture and not a quirk. But the paper is short, hardware-shaped, and is the one item in my block with no free copy anywhere I could find; a recording of the talk is public if you want the content without the paper.

*Cited in the preprint for:* Cited as an e-prop port to SpiNNaker 1, and specifically as one of the two ports that incorporate event-driven weight updates; also noted as the only port that reproduces the pattern-generation task and one of two addressing evidence accumulation.

*Readable:* paywalled (free version: a public recording of the conference talk at neuropac.info/video/online-learning-in-snns-with-e-prop-and-neuromorphic-hardware-adam-perrett-2022/) — https://doi.org/10.1145/3517343.3517352

## [68] — 0.1

**Oliver Rhodes, Luca Peres, Andrew G. D. Rowley, Andrew Gait, Luis A. Plana, Christian Brenninkmeijer and Steve B. Furber (2019), "Real-time cortical simulation on neuromorphic hardware", Philosophical Transactions of the Royal Society A 378.2164, p. 20190160. doi: 10.1098/rsta.2019.0160**

A pure benchmarking paper: the Manchester group runs the standard cortical microcircuit model - about 77,000 neurons and 300 million synapses - on SpiNNaker 1 and shows it executes ten seconds of biological time in ten seconds of wall-clock time, where conventional supercomputer simulators were three times slower and GPUs twice as slow, at roughly a tenth of the energy. They also leave it running for twelve-hour stretches to prove the machine is stable. There is no learning anywhere in this paper - the microcircuit is a fixed-weight network and the question is purely whether the hardware can keep up. The preprint cites it only to establish what SpiNNaker 1 is, next to the e-prop port in reference 67. For walnutbutter it is the least relevant item in my block: you are not running on neuromorphic hardware, your bottleneck is core count on a desktop, and their real-time claim is about a model you are not simulating. Skip it unless you find yourself curious about how a fully event-driven machine handles 300 million synapses.

*Cited in the preprint for:* Cited alongside reference 67 purely to identify the SpiNNaker 1 platform on which that e-prop port runs.

*Readable:* open access — https://pmc.ncbi.nlm.nih.gov/articles/PMC6939236/

## [69] — 0.3

**Amirhossein Rostami, Bernhard Vogginger, Yexin Yan and Christian G. Mayr (2022), "E-prop on SpiNNaker 2: Exploring online learning in spiking RNNs on neuromorphic hardware", Frontiers in Neuroscience 16, p. 1018006. doi: 10.3389/fnins.2022.1018006**

The Dresden group implements e-prop on a prototype of SpiNNaker 2 and trains a small spiking recurrent network from scratch, in real time, on Google Speech Commands keyword spotting: 91.1 percent accuracy with 25,000 weights in 680 kilobytes of memory, which they argue is far leaner than comparable spiking approaches and about twelve times cheaper in energy than training the same thing on a V100. The genuinely interesting part is not the accuracy but the profiling: they measure where e-prop's memory and time actually go, and use that to argue about when e-prop beats backpropagation-through-time at the edge, and they are the only port in this group that explicitly measures how processing time scales with network size. Like reference 67, they apply weight updates on spikes rather than on a clock. For you this is a second, better-documented data point that a spike-triggered three-factor update is practical, plus a rare honest accounting of what an eligibility trace costs per synapse - which bears on your open question about whether the exploring object and the credited object should both live at the synapse. But it is still a port paper: the rule is taken as given and bent to fit ARM cores, not interrogated. Free at Frontiers.

*Cited in the preprint for:* Cited as the e-prop port to SpiNNaker 2, as the second port using event-driven weight updates, as the only study explicitly comparing learning performance to the original, and as the only one analysing scaling of processing time against network size.

*Readable:* open access — https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2022.1018006/full

## [70] — 0.4

**Charlotte Frenkel and Giacomo Indiveri (2022), "ReckOn: A 28nm Sub-mm2 Task-Agnostic Spiking Recurrent Neural Network Processor Enabling On-Chip Learning over Second-Long Timescales", 2022 IEEE International Solid-State Circuits Conference (ISSCC), Vol. 65, pp. 1-3. doi: 10.1109/ISSCC42614.2022.9731734**

ReckOn is a half-square-millimetre chip that learns on itself, over timescales of seconds, in under 150 microwatts, demonstrated on hand gestures from a spiking camera, spoken digits from a spiking cochlea, and a delayed-supervision navigation task where the reward arrives long after the decisions that earned it. It is a circuits paper, but it contains one argument that is squarely about learning-rule design and is the reason I scored it above the other hardware ports. E-prop's eligibility trace, as Bellec wrote it, needs multi-timescale filtering inside every synapse, which is unaffordable on chip; Frenkel and Indiveri show you can instead lengthen the neuron's leak time constant to match the task, simplify the e-prop equations for plain integrate-and-fire neurons, and end up with an update that cleanly separates into a presynaptic eligibility term and postsynaptic learning-signal and slope terms. The consequence is that eligibility storage scales with the number of neurons rather than the number of synapses. That is worth your attention precisely because it runs opposite to where you are heading: you are asking how to move the exploration down to the synapse so the exploring thing and the credited thing are the same thing, and this paper is a careful, quantified account of what you buy by moving credit the other way, up to the neuron. The evidence-accumulation-style navigation task is also the closest thing in this block to your own problem. Free preprint on arXiv.

*Cited in the preprint for:* Cited as an e-prop adaptation to neuromorphic hardware, as an example of simplifying the algorithm to meet hardware constraints, and as one of two ports addressing evidence accumulation; also for its spiking Heidelberg digits and synthetic behavioural datasets.

*Readable:* preprint — https://arxiv.org/abs/2208.09759

## [71] — 0.2

**Johanna Senk et al. (2025), "Constructive community race: full-density spiking neural network model drives neuromorphic computing", arXiv:2505.21185. doi: 10.48550/arXiv.2505.21185**

This is a review written jointly by most of the European neuromorphic groups - SpiNNaker, BrainScaleS, GeNN, NEST - about a single shared benchmark. In 2014 a model was published representing every neuron and synapse under a square millimetre of cortex at full density, which mattered because it removed the guesswork involved in scaling a model down. It was too expensive to run, which turned it into a de facto competition; within a few years several platforms reached and beat real time on it at much lower energy. The paper reviews how each technology got there and then draws lessons for designing the next generation of benchmarks. There is no learning rule in it and no mechanism you could adopt. I scored it low for that reason, but there is one thread that touches you: the paper's central point is that full-density models behave differently from downscaled ones, and that larger networks are less densely connected. You have a hazard that shrinks as the square root of the network size, which is your own answer to the same question - how do you keep a network behaving itself as it grows. Reading how a whole community argued about density and scale might sharpen how you defend that choice, but it will not give you a number. Free on arXiv.

*Cited in the preprint for:* Cited in the closing discussion as a recent effort at comparing the time and energy demands of neuromorphic and conventional solutions on the same task at fixed accuracy.

*Readable:* preprint — https://arxiv.org/abs/2505.21185

## [72] — 0.5

**Rodney J. Douglas, Kevan A. C. Martin and David Whitteridge (1989), "A Canonical Microcircuit for Neocortex", Neural Computation 1.4, pp. 480-488. doi: 10.1162/neco.1989.1.4.480**

This is the old classic of the block and the only one written by people looking down a microscope. Douglas, Martin and Whitteridge took anatomy from individually filled neurons in cat visual cortex plus intracellular recordings from live animals and distilled the whole thing into the simplest circuit that could reproduce what they measured: three populations - pyramidal cells driven by thalamus, pyramidal cells driven intracortically, and inhibitory smooth cells. It is eight pages long and it states three findings that they aimed straight at computational modellers. First, excitation and inhibition are not separable events; any activation of cortex sets both in motion in every neuron. Second, the thalamic input is not the main excitation any neuron receives - most of it comes from other cortical cells. Third, and this is the one for you, the time course of excitation and inhibition is far longer than the synaptic delays involved, so cortical processing cannot depend on precise timing between individual synaptic inputs. Your neurons are no-leak evidence accumulators with discrete hop delays and a shaping function of spike timing, and this paper is a thirty-five-year-old empirical argument about exactly how much timing precision the real thing can and cannot be relying on. The preprint cites it only for scale - the microcircuit has a hundred times more neurons than any e-prop model - but the paper is worth more than that citation gives it. Free PDF on the authors' own institute server in Zurich; the MIT Press original is paywalled.

*Cited in the preprint for:* Cited, with references 73 and 74, to establish that the basic computational unit of mammalian cortex contains two orders of magnitude more neurons than present-day e-prop network models, motivating the scaling study.

*Readable:* paywalled (free version: author copy on the Institute of Neuroinformatics server, University of Zurich) — https://services.ini.uzh.ch/admin/extras/doc_get.php?id=42860

## [73] — 0.4

**Braitenberg & Schüz (1998), "Cortex: Statistics and Geometry of Neuronal Connectivity", Springer Berlin Heidelberg (2nd edn), doi:10.1007/978-3-662-03733-1**

This is the classic book of cortical bookkeeping. Braitenberg and Schüz spent about twenty years counting things in mouse cortex — how many neurons per cubic millimetre, how much of the tissue is axon versus dendrite versus synapse, how long the average axon is, how many synapses a pyramidal cell makes and receives — and then assembled those counts into a single coherent picture of the cortical network. Their conclusion is that cortex is, to a first approximation, a diffuse and fairly homogeneous network of excitatory pyramidal cells wired more or less at random over long distances, and that this is exactly the architecture an associative memory or a Hebbian cell-assembly machine would have. It is where the numbers everyone quotes come from: roughly ten thousand synapses per neuron, connection probabilities of a percent or so, sparse rather than dense wiring. For you it is background rather than machinery: it is the empirical source of the sparse-wiring assumption your engines already make, and it is the reason nobody in this field builds all-to-all networks, but it says nothing about learning rules, stochastic firing, or credit assignment, and it will not settle any open question on your list. Read it if you want to know where the in-degree numbers came from; skip it if you are chasing the learning rule. It is also a paywalled Springer book with no legitimate free copy, which given you are reading top-down makes it an easy one to pass over.

*Cited in the preprint for:* One of three citations (72–74) backing the claim that the cortical microcircuit already contains two orders of magnitude more neurons than present-day e-prop models, motivating the paper's scaling study.

*Readable:* paywalled (free version: none found — the same connectivity statistics are quoted openly in refs [74] and [80]) — https://link.springer.com/book/10.1007/978-3-662-03733-1

## [74] — 0.5

**Potjans & Diesmann (2014), "The cell-type specific cortical microcircuit: relating structure and activity in a full-scale spiking network model", Cerebral Cortex 24(3):785–806, doi:10.1093/cercor/bhs358**

Potjans and Diesmann built the reference spiking model of one square millimetre of cortex: about 80,000 neurons and 300 million synapses, split into eight populations (excitatory and inhibitory cells in layers 2/3, 4, 5 and 6). Their real contribution is the connectivity map — they reconciled two incompatible bodies of experimental data, anatomical reconstructions of overlapping axons and dendrites on one side and paired patch-clamp recordings on the other, by modelling the lateral spread of connections and correcting for how each method samples. When they ran it, the spontaneous activity came out asynchronous and irregular with layer-specific firing rates matching in-vivo recordings, including the famously low rates of layer 2/3 excitatory cells that earlier models could never reproduce. For you this is the concrete version of what reference [73] states as counts: a network specified precisely enough that you could write it down in AUTHORITY.md, with in-degrees in the thousands and a sparse permissive wiring scheme. It is worth knowing because it is the standard against which "natural density" is measured in the whole Jülich/NEST line of work that produced this preprint, and because it shows that a network of fixed in-degree sits in a stable asynchronous regime — the regime your hazard scaling is trying to hold onto as N grows. But it has no learning in it at all, so it will not change your rule.

*Cited in the preprint for:* Cited with [72] and [73] to establish the size of the cortical microcircuit that present-day e-prop network models fall two orders of magnitude short of.

*Readable:* open access — https://pmc.ncbi.nlm.nih.gov/articles/PMC3920768/

## [75] — 0.9

**Lillicrap, Cownden, Tweed & Akerman (2016), "Random synaptic feedback weights support error backpropagation for deep learning", Nature Communications 7:13276, doi:10.1038/ncomms13276**

This is the feedback-alignment paper, and it is one of the genuinely surprising results in the field. Backpropagation requires that the pathway carrying the error backwards use exactly the transpose of the forward weights — every synapse would have to know the strength of a different synapse somewhere else, which no brain can arrange. Lillicrap and colleagues simply replaced the backward matrix with fixed random numbers and trained anyway. It works: the forward weights rotate over training until they come into rough alignment with the random feedback, so the random matrix starts delivering a signal that is within ninety degrees of the true gradient, and learning proceeds about as well as with real backpropagation. The feedback never learns; the forward path learns how to make it useful. This matters to you directly because it is the licence the whole three-factor literature relies on when it says a synapse can be told about error by a broadcast signal it did not help compute — it is exactly the step that lets e-prop's learning signal be local. Your own rule goes further still: a single global scalar is the extreme case of a broadcast signal, and although your justification for it is REINFORCE rather than alignment, this is the paper that made the field comfortable with credit arriving from a channel that is not the gradient's own. It is short, free, and it is the ancestor of references [76] and [77] below, so read it first and the other two become quick.

*Cited in the preprint for:* Cited as the idea e-prop follows when it replaces the transposed output weight matrix by a fixed random feedback matrix B, avoiding the biologically implausible weight symmetry that backpropagation-through-time requires.

*Readable:* open access — https://pmc.ncbi.nlm.nih.gov/articles/PMC5105169/

## [76] — 0.6

**Nøkland (2016), "Direct Feedback Alignment Provides Learning in Deep Neural Networks", Advances in Neural Information Processing Systems (NeurIPS) 29**

Nøkland took feedback alignment one step further. In the original scheme the error still travels layer by layer, just through random weights; here the error is delivered from the output layer directly to every hidden layer at once, each through its own fixed random matrix, with no backward chain at all. He shows this still reaches zero training error, in convolutional and very deep networks, and gets within a whisker of backpropagation on MNIST and CIFAR — 1.45 percent error on permutation-invariant MNIST when combined with dropout. The point he is making is a locality point: the error signal is now almost local to each layer, there is no backward pass to sequence, and nothing needs to know the forward weights. For you this is closer to home than reference [75], because it is the first demonstration that a signal simply broadcast from the output, with no structure relating it to the forward path, is enough to train a deep network — which is the same shape as your global dopamine scalar, only vector-valued. The honest caveat is that the preprint itself points out the distinction dissolves in a single-layer network, and walnutbutter's credit is one number, not a matrix, so nothing here changes a decision you have open. Read it as the natural sequel to [75], for the intuition about how little structure the feedback channel actually needs.

*Cited in the preprint for:* Cited to distinguish direct feedback alignment — the output layer sends the error through a separate random matrix straight to each hidden layer — from plain feedback alignment [75] and broadcast alignment [77]; the preprint notes all three coincide in the single-hidden-layer networks it uses.

*Readable:* proceedings (free) — https://proceedings.neurips.cc/paper/2016/hash/d490d7b4576290fa60eb31b5fc917ad1-Abstract.html

## [77] — 0.7

**Samadi, Lillicrap & Tweed (2017), "Deep Learning with Dynamic Spiking Neurons and Fixed Feedback Weights", Neural Computation 29(3):578–602, doi:10.1162/NECO_a_00929**

This is the paper that carried feedback alignment across into spiking neurons, and it is the closest of this trio to what you are building. Samadi, Lillicrap and Tweed name three things that separate real neurons from the units in a deep network: they emit spikes rather than graded values, their input-output relation is dynamic rather than a fixed smooth function, and there is no known way for a neuron to learn the strength of synapses elsewhere in the circuit. They answer all three at once — approximate the dynamic, non-differentiable spiking response with a piecewise-smooth surrogate so a derivative exists, and use a single fixed random matrix broadcast from the output to every hidden layer so no weight transport is needed. The network is leaky integrate-and-fire with Bernoulli-encoded pixel inputs, and it reaches roughly 96 percent on MNIST. So: spiking units, a broadcast global error, a surrogate for the spike, MNIST — four of your own ingredients in one 2017 paper, which makes it a useful point of comparison even though their surrogate does the job your escape hazard does for free (your firing probability is already smooth in the margin, so you have an exact score function and never need to invent a derivative). Worth reading for that contrast alone. The sting is that it is paywalled at MIT Press with no repository copy anywhere I could find, so you may get only the abstract unless you want to write to Tweed.

*Cited in the preprint for:* Cited as "broadcast alignment" — the variant in which one and the same random feedback matrix is used for all hidden layers — alongside [75] and [76]; the preprint notes the three are equivalent for its single-layer networks.

*Readable:* paywalled (free version: abstract only, on PubMed) — https://pubmed.ncbi.nlm.nih.gov/28095195/

## [78] — 0.8

**van Albada, Helias & Diesmann (2015), "Scalability of Asynchronous Networks Is Limited by One-to-One Mapping between Effective Connectivity and Correlations", PLOS Computational Biology 11(9):e1004490, doi:10.1371/journal.pcbi.1004490**

This paper is about what happens to a spiking network's statistics when you change its size, and it reaches a sharp negative result. Van Albada, Helias and Diesmann show that in an asynchronous network — the irregular, low-rate regime cortex sits in — the effective connectivity at the population level determines the temporal structure of pairwise correlations one-to-one. You can shrink a network and keep the mean firing rates right by rescaling the synaptic weights, but you cannot keep the rates and the correlations right at the same time except within a narrow window, and how wide that window is depends on how much variance the external input supplies. Beyond it the network tips into a different regime, typically high-rate or synchronous. That is precisely your territory: your kappa(N) = sqrt(60/N) factor is a rule for changing the firing statistics as N changes, chosen to keep the network in a workable regime, and this paper is the careful analysis of what is and is not preservable when you do that. It will not hand you the right exponent — their setting is a balanced network with no learning, and your hazard is a per-neuron escape rate rather than a synaptic weight — but it tells you which quantities you are trading against each other, and gives you the language (effective connectivity, asynchronous irregular state) to argue about whether sqrt is the right scaling or merely a serviceable one. Free on arXiv and at PLOS.

*Cited in the preprint for:* Cited in the scaling experiments as the reason to avoid the network falling into a high-activity or high-synchrony state, which would make the timing comparisons across network sizes unfair.

*Readable:* open access — https://arxiv.org/abs/1411.4770

## [79] — 0.3

**Lansner & Diesmann (2012), "Virtues, Pitfalls, and Methodology of Neuronal Network Modeling and Simulations on Supercomputers", in Computational Systems Neurobiology, Springer Dordrecht, pp. 283–315, doi:10.1007/978-94-007-3858-4_10**

This is a review chapter on how to do large-scale spiking network simulation on supercomputers properly: what these models are for, what they can and cannot tell you, how to set up parallel simulations, where the common methodological traps are, and how choices made for computational convenience quietly become modelling assumptions. The preprint leans on it for exactly that last point twice over — once for the fact that keeping the number of synapses per neuron fixed as the network grows holds the firing rate roughly steady (which is why their weak-scaling runs are a fair comparison), and once, more pointedly, to name the historical convention of tying every transmission delay to the simulator's time step as a convenience rather than a biological choice, which is what the paper's generalized delays undo. That second use touches you: walnutbutter's hop delays are discrete integers for the same convenience, and the preprint's move away from that is on your horizon. But this chapter is a methodology review of running simulators at scale, not a source of mechanism, and it is a paywalled Springer book chapter with no repository copy anywhere — KTH's own archive lists it with nothing but the publisher link. Low score on both counts: little in it for your rule, and you cannot read it for free anyway.

*Cited in the preprint for:* Cited twice: with [18] for the point that holding in-degree constant as the network grows keeps the firing rate roughly constant while correlations fall, and in Appendix B as the "historical method" of coupling transmission delays to the simulation's temporal resolution — the practice the paper's generalized delays are introduced to replace.

*Readable:* paywalled (free version: none found — listed without full text in KTH's DiVA repository) — https://link.springer.com/chapter/10.1007/978-94-007-3858-4_10

## [80] — 0.5

**Senk, Kriener, Djurfeldt, Voges, Jiang, Schüttler, Gramelsberger, Diesmann, Plesser & van Albada (2022), "Connectivity concepts in neuronal network modeling", PLOS Computational Biology 18(9):e1010086, doi:10.1371/journal.pcbi.1010086**

Senk and colleagues went through a large number of published network models, from ModelDB and Open Source Brain and the literature, and found that a substantial fraction of the connectivity descriptions are ambiguous: the text does not say enough for anyone to rebuild the same network, and different readers implement different things. Their response is a taxonomy and a standard — precise definitions of every common connection rule, from fully deterministic to various probabilistic schemes, mathematical statements of what each one actually generates, the vocabulary for the edge cases (autapses, multapses, fixed in-degree versus fixed total number versus per-pair Bernoulli), and a graphical notation for drawing a network's connectivity unambiguously. The preprint uses it only in passing, for the two words autapse and multapse. For you it is more useful than that single citation suggests. You have a specification file whose whole purpose is that the three engines build the same network, and your stated value is permissive structures — a neuron may sit in several zones, connections are not artificially restricted — which means the questions this paper standardizes (may a neuron connect to itself, may a pair share two synapses, is the in-degree fixed or drawn) are live clauses in AUTHORITY.md rather than trivia. This will not change your learning rule, but it is the right thing to have open beside you the next time you write down how wiring is generated. Free at PLOS and on arXiv.

*Cited in the preprint for:* Cited for the definitions of autapses and multapses — self-connections and multiple synapses between the same pair of neurons — which the scaling experiments' recurrent connectivity excludes.

*Readable:* open access — https://arxiv.org/abs/2110.02883

## [81] — 1.0

**Friedemann Zenke and Surya Ganguli (2018), "SuperSpike: Supervised Learning in Multilayer Spiking Neural Networks", Neural Computation 30(6):1514-1541, doi:10.1162/neco_a_01086**

Zenke and Ganguli take a network of ordinary deterministic integrate-and-fire neurons, where the spike is a hard all-or-nothing event with no derivative, and derive a learning rule that trains it anyway. Their trick is to write down the gradient of a spike-train error as if the neuron's output were a smooth function of its membrane voltage, substituting a bump-shaped 'surrogate' for the missing derivative, and the rule that falls out has exactly three multiplicative pieces: a filtered trace of presynaptic activity, that surrogate function of the postsynaptic neuron's own voltage, and an error signal arriving from outside the synapse. The first two together they call the eligibility trace and suggest a calcium transient could hold it; the third is the third factor. They then ask how hidden units are supposed to learn at all without backpropagation, and test three answers - feedback along the transposed forward weights, feedback along fixed random weights, and one single global broadcast signal shared by every neuron - finding all three work on easy tasks and only the symmetric one holds up on hard ones. This is the paper that made 'surrogate gradient plus three-factor rule' a going concern in spiking networks, and e-prop is its direct descendant. For Byron it is a must-read for two separate reasons: it is the cleanest derivation of the eligibility-times-global-signal structure walnutbutter already has, and the uniform-feedback experiment is the honest published answer to how far one global scalar can carry a spiking network before it stops working. Note also the interesting contrast - Zenke's neurons are deterministic and the smoothness is imported by hand as a fiction, where walnutbutter's neurons are stochastic and the smoothness is real, which means Byron's eligibility is an exact score function rather than a surrogate.

*Cited in the preprint for:* Cited only in supplementary Figure S6, as the source of the 'fast sigmoid derivative' surrogate gradient shape they compare against their own piecewise-linear and exponential surrogates.

*Readable:* open access — https://arxiv.org/abs/1705.11146

## [82] — 0.3

**Wei Fang, Zhaofei Yu, Yanqi Chen, Tiejun Huang, Timothée Masquelier and Yonghong Tian (2021), "Deep Residual Learning in Spiking Neural Networks", Advances in Neural Information Processing Systems (NeurIPS) 34:21056-21069**

This is a deep-learning architecture paper. The problem it solves is that if you take a ResNet - the standard trick of letting each layer learn a correction on top of a shortcut that passes the signal through unchanged - and naively swap the activation functions for spiking neurons, the shortcut stops being an identity and gradients vanish or explode as you stack layers. Fang and colleagues propose SEW-ResNet, which combines the shortcut and the residual branch with an element-wise operation on the spikes themselves rather than on the voltages, prove this genuinely implements identity mapping, and use it to train spiking networks more than a hundred layers deep for the first time, beating the previous best directly-trained spiking results on ImageNet and DVS benchmarks. It matters to the e-prop preprint for one incidental reason: the arctangent-shaped surrogate gradient this group popularized is one of four shapes the preprint's supplementary figure compares. For Byron it is honestly peripheral. He is not stacking layers and has no shortcut connections, and the paper's contribution is architectural rather than anything about credit assignment or noise. The one thing worth taking from it is that the arctangent surrogate exists and is widely used, which is a small data point for his open question about what shape the firing hazard should have - but he would get that from the preprint's own figure without reading this.

*Cited in the preprint for:* Cited only in supplementary Figure S6, as the source of the inverse-tangent (arctangent) surrogate gradient shape they include in their comparison of surrogate functions.

*Readable:* proceedings (free) — https://proceedings.neurips.cc/paper/2021/hash/afe434653a898da20044041262b3ac74-Abstract.html

## [83] — 1.0

**Eugene M. Izhikevich (2007), "Solving the distal reward problem through linkage of STDP and dopamine signaling", Cerebral Cortex 17(10):2443-2452, doi:10.1093/cercor/bhl152**

This paper is already in Byron's references folder, and it is the one that names the problem every rule of walnutbutter's kind has to solve. The setup: a reward arrives seconds after the action that earned it, and in those intervening seconds every neuron and every synapse in the network has gone on firing. How does the reward find the handful of synapses that actually deserve it? Izhikevich's answer is that a coincidence of pre- and postsynaptic spikes does not change the weight directly; it sets a slowly decaying flag at that synapse - an eligibility trace, with a lifetime of a few seconds - and the weight only changes when a global rise in extracellular dopamine multiplies against whatever flags are still lit. The crucial demonstration is the negative one: he shows that the random firing going on during the waiting period does not corrupt the trace, because the trace is set by precise spike coincidences rather than by rate, so the network stays deaf to the noise it is swimming in. Earlier theoretical treatments had quietly assumed the network was silent or frozen while waiting, and he is explicit that this is what distinguishes his account from theirs. For Byron this is the direct ancestor of his dopamine rule - eligibility at the synapse, one global scalar broadcast to all of them, weight change only at the product - and the noise-immunity argument is exactly the question he should be asking about his own eligibility under continuous free running.

*Cited in the preprint for:* Cited together with [84] for the 'distal reward problem' - the preprint's justification for adding a transmission delay between a synapse's activity and the arrival of the learning signal that grades it.

*Readable:* in references/ already — https://www.izhikevich.org/publications/dastdp.pdf

## [84] — 0.9

**Wiebke Potjans, Markus Diesmann and Abigail Morrison (2011), "An Imperfect Dopaminergic Error Signal Can Drive Temporal-Difference Learning", PLOS Computational Biology 7(5):e1001133, doi:10.1371/journal.pcbi.1001133**

Potjans and colleagues build an actor-critic reinforcement learner entirely out of spiking neurons - cortex encodes the state, striatum and a population of dopaminergic neurons form the critic, a separate group forms the actor - and ask whether it can still learn when the dopamine signal is not the clean plus-or-minus number that the textbook temporal-difference algorithm assumes. Real dopaminergic neurons fire at a low baseline rate, so they can signal a large positive surprise by firing hard but can only signal a negative surprise by falling silent, which caps how much negative information the signal can carry. Each synapse in their model keeps two variables, a presynaptic activity trace standing in for firing rate and a presynaptic efficacy trace that resets on each spike, and their product opens a narrow plasticity window just after a state transition. The result is a clean asymmetry: on tasks with sparse positive rewards the spiking network learns as well as the abstract algorithm, on mixed positive and negative rewards it degrades badly, and on purely negative rewards it fails outright. This is directly about the thing Byron's dopamine rule is - a single global scalar multiplying a per-synapse eligibility - and it is the best published warning about what a globally broadcast reward signal can and cannot encode. It scores just below the top only because the temporal-difference and actor-critic machinery around it is not what walnutbutter does; the lesson about signal asymmetry is what he should take.

*Cited in the preprint for:* Cited alongside [83] for the distal reward problem, as the authors' justification for modelling a delay between synaptic activity and the arrival of the error-carrying learning signal.

*Readable:* open access — https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1001133

## [85] — 0.8

**Zuzanna Brzosko, Susanna B. Mierau and Ole Paulsen (2019), "Neuromodulation of Spike-Timing-Dependent Plasticity: Past, Present, and Future", Neuron 103(4):563-581, doi:10.1016/j.neuron.2019.05.041**

This is a Neuron review of the experimental literature behind the third factor. Its starting point is a mismatch nobody has satisfactorily closed: spike-timing-dependent plasticity works on a millisecond window, behavioural learning works on seconds to minutes, and the two timescales do not obviously meet. The authors argue neuromodulation is the bridge, and then survey what is actually measured at the bench - acetylcholine, dopamine, noradrenaline, serotonin and others changing not just the size but the sign and the timing window of spike-timing-dependent plasticity - with the underlying receptor and second-messenger mechanisms, the functional consequences, and the disorders in which the modulation goes wrong. It is a review, not new data, but it is the review a modeller should read before asserting that a global chemical signal gating a synaptic trace is biologically real. For Byron it sits one shelf over from Fremaux and Gerstner, which he already has: Fremaux and Gerstner give the theory of three-factor rules, this gives the experiments they rest on, and in particular it gives the honest range of timescales over which a neuromodulatory third factor actually arrives. That bears on his open question about the teacher signal and on whether a delayed global signal is a compromise or the biological default. It will not change a line of his code.

*Cited in the preprint for:* Cited as the evidence that the learning signal may travel by slow neuromodulatory processes and chemical intermediaries, i.e. that a delay of the third factor is biologically expected rather than an artefact.

*Readable:* paywalled (free version: author accepted manuscript in the Cambridge repository) — https://www.repository.cam.ac.uk/handle/1810/293509

## [86] — 0.2

**Caio Seguin, Olaf Sporns and Andrew Zalesky (2023), "Brain network communication: concepts, models and applications", Nature Reviews Neuroscience 24(9):557-574, doi:10.1038/s41583-023-00718-5**

This is a review from network neuroscience, the field that treats the brain as a graph of regions and fibre tracts and asks how a signal gets from one region to another. Its central argument is that the long-standing assumption - that signals travel along shortest paths, as if every neuron knew the global map - is implausible, and that a whole family of alternative communication models has grown up in its place: diffusion and random walks, navigation using spatial position, broadcast, and various hybrids. The authors lay out the link between the graph mathematics and the biology it is meant to stand for, notably transmission delay and metabolic cost, organize the competing models and their measures into a taxonomy so a newcomer can tell them apart, show how these models have been applied in basic, cognitive and clinical work, and close with recommendations for validating them. It is a good review of a real field, but it operates at the scale of brain regions and diffusion-imaging tractography, which is several levels of abstraction above anything in walnutbutter. Byron has discrete hop delays between neurons and permissive sparse wiring, and nothing here speaks to either at the level he works: the delays in this review are centimetres of white matter, not his integer hops. The preprint itself uses it for a single throwaway clause. Skip unless he becomes curious about connectome-scale modelling for its own sake.

*Cited in the preprint for:* Cited for the single claim that neural circuits are spatially distributed, so physical distance between populations lengthens signal travel time - a one-clause justification for transmission delays.

*Readable:* paywalled — https://www.nature.com/articles/s41583-023-00718-5

## [87] — 0.8

**Guillermo Martín-Sánchez, Sander Bohté and Sebastian Otte (2022), "A Taxonomy of Recurrent Learning Rules", International Conference on Artificial Neural Networks (ICANN), Springer LNCS 13529:478-490, doi:10.1007/978-3-031-15919-0_40**

This short conference paper is the map of the territory that e-prop lives in. It starts from backpropagation through time, the standard way of training a recurrent network, and points out its two sins: it is non-causal, because computing the update for a moment requires knowing what happens afterwards, and it is non-local, because the error has to be carried across every neuron and synapse in the network. Real-time recurrent learning fixes causality by carrying a recursively updated trace forward instead of looking backward, at ruinous cost, and remains non-local. The authors rederive real-time recurrent learning from backpropagation through time in careful notation so the relationship is visible, then show precisely which term e-prop throws away - the explicit recurrent dependence between different neurons - to buy locality and causality. The pretty result is a family they call m-order e-prop: at m equal to one you have e-prop, fully local and causal and maximally approximate; at m equal to the full sequence length you have the exact gradient; in between you buy accuracy by looking m steps ahead. For Byron this is the clearest short statement of what 'local', 'causal' and 'online' actually mean as technical claims, which is the vocabulary he will need to say what walnutbutter's rule is and is not. It is one step removed rather than a must-read, because his rule is a REINFORCE-style score function rather than an approximation to a loss gradient, so he sits outside this family rather than in it - but knowing where the family is tells him what he is outside of.

*Cited in the preprint for:* Used substantively, in two places: the preprint borrows this paper's 'readout-restricted derivative' notation to do its delayed-credit-assignment algebra, and Appendix F leans on its framework to classify e-prop as local, causal and online and to place it against BPTT and RTRL.

*Readable:* preprint — https://arxiv.org/abs/2207.11439

## [88] — 0.2

**Rotem Zamir Aviv, Ido Hakimi, Assaf Schuster and Kfir Yehuda Levy (2021), "Asynchronous distributed learning: Adapting to gradient delays without prior knowledge", Proceedings of the 38th International Conference on Machine Learning (ICML), PMLR 139:436-445**

This is a machine-learning optimization theory paper about training on a cluster. When many machines compute gradients in parallel against a shared set of weights, each gradient arrives stale - computed against weights that have since moved - and the standard convergence proofs need to know in advance how stale, plus how smooth the objective is and how noisy the gradients are, none of which anyone actually knows and all of which drift as machines are reallocated. The authors give a method for constrained stochastic convex optimization that adapts to the delays as it goes and carries non-asymptotic convergence guarantees requiring no advance knowledge of delay, smoothness or gradient variance. It is careful work and the guarantees are real. Its bearing on walnutbutter is thin: the preprint cites it purely to name-check the fact that applying a gradient later than you computed it is a known and studied thing, which is the position their delayed learning signal puts them in. Byron's late-signal rule is the same shape of situation, so the connection is not nothing - but the theory here is convex optimization with a synchronization budget, and it will not tell him anything about what a delayed dopamine signal does to a spiking network. Read the one sentence in the preprint that cites it, not the paper.

*Cited in the preprint for:* Cited for the name and the existence of the technique: that computing a gradient at one moment and applying it later is known as delayed gradient descent, standard in distributed and asynchronous optimization.

*Readable:* proceedings (free) — https://proceedings.mlr.press/v139/aviv21a.html

## [89] — 0.3

**Deng, Shen, Li, Sun, Li & Tao (2025), "Toward Understanding the Generalizability of Delayed Stochastic Gradient Descent", IEEE Transactions on Pattern Analysis and Machine Intelligence 47(9), 7976–7986, doi:10.1109/tpami.2025.3572251**

This is a piece of machine-learning optimization theory about what happens when you train a model on many machines at once and the gradient each worker applies is stale — computed from weights that have since moved on. Everyone assumed staleness was purely a cost. The authors prove otherwise: using a stability argument, they get much tighter bounds on generalization error for delayed SGD with a delay of tau, and the bounds shrink as tau grows. In plain terms, a late gradient acts a bit like noise injected into the optimizer and makes the trained model generalize better rather than worse. The e-prop preprint reaches for it at exactly one point — having just shown that in their scheme the eligibility trace is multiplied by a learning signal that arrives some milliseconds later, they cite this to argue the delay is not a defect to be engineered away. For you that argument is the interesting part, not the paper: walnutbutter's reward signal is also global and also late relative to the spike it is scoring, and this is a published reason to stop treating that as an approximation you are tolerating. The paper itself, though, is convex and strongly-convex quadratic analysis of mini-batch SGD on a cluster, which is a long way from a hazard function and a dopamine scalar. Read the abstract and the one-sentence conclusion; you will not need the proofs.

*Cited in the preprint for:* Cited once (Sec. on learning-signal delay) to argue that the delay between spike and arriving learning signal is not harmful, since asynchronous delays have been shown to reduce generalization error.

*Readable:* paywalled (free version: arXiv preprint, v4, same content) — https://arxiv.org/abs/2308.09430

## [90] — 0.5

**Brette, Rudolph, Carnevale, Hines, Beeman, Bower, Diesmann, Morrison, Goodman, Harris et al. (2007), "Simulation of networks of spiking neurons: a review of tools and strategies", Journal of Computational Neuroscience 23, 349–398, doi:10.1007/s10827-007-0038-6**

This is the standing review of how people actually build spiking network simulators, written jointly by the authors of most of them (NEURON, NEST, GENESIS, Brian, and others) after a workshop. The first half is the part worth your time: it lays out the two strategies for advancing a simulation — clock-driven, where you step every neuron on a global time grid, versus event-driven, where you only touch a neuron when a spike arrives and you integrate analytically across the gap — and it is honest about the errors each one commits, including the fact that a time grid quantizes spike times and can change network behaviour systematically rather than just adding jitter. The second half is a simulator-by-simulator tour with benchmarks, now nineteen years old and mostly of historical interest. The preprint cites it only in its architecture appendix, as the reference for the modular design their NEST code has to fit into. For you it earns a middling score on its own merits, not on how the preprint uses it: you have written three engines from scratch on discrete hops and you are about to be told, by the rest of this bibliography, that event-driven updating is the thing that makes e-prop scale. This is the document that explains what that division actually is and what each side costs, in ordinary language, before anyone starts talking about eligibility traces. Nothing in it will change a rule in AUTHORITY.md.

*Cited in the preprint for:* Cited in the architecture appendix as the reference for the modular simulator design their reference implementation had to remain compatible with.

*Readable:* open access — https://arxiv.org/abs/q-bio/0611089

## [91] — 0.6

**Clopath, Büsing, Vasilaki & Gerstner (2010), "Connectivity reflects coding: a model of voltage-based STDP with homeostasis", Nature Neuroscience 13(3), 344–352, doi:10.1038/nn.2479**

Clopath and Gerstner replace spike-timing-dependent plasticity with something simpler and more fundamental: a synapse changes according to presynaptic spike arrival paired with the postsynaptic membrane voltage, where that voltage is read through two filters with different time constants — a fast one for potentiation and a slow one for depression — plus a homeostatic term that holds the postsynaptic firing rate in range. From that one rule they recover the standard timing curve, the nonlinear triplet and quadruplet effects, the voltage dependence, and the frequency dependence, none of which were put in by hand. They then run it in a recurrent network and get the paper's headline: the connectivity that emerges tells you what code the network is using — temporally correlated input grows strong one-way chains, rate-coded input with only spatial correlation grows strong bidirectional pairs. The preprint cites it, together with Urbanczik–Senn, as one of the two plasticity models NEST had already converted to event-driven form, which is the engineering precedent for their EpropArchivingNode. Its value to you is as ancestry rather than machinery — this is the same laboratory whose escape-noise neuron you are using, and it is the cleanest demonstration that a rule reading a continuous postsynaptic quantity can explain the spike-timing data people usually treat as primary. Your rule is three-factor and global-reward, not voltage-based and unsupervised, so nothing here transfers directly; read it to know the territory and to see why voltage-dependent rules are the hard case for event-driven updating.

*Cited in the preprint for:* Cited alongside [92] as one of the two voltage-based plasticity models previously re-implemented in an event-driven NEST framework (ClopathArchivingNode), the precedent for the authors' own EpropArchivingNode.

*Readable:* paywalled (free version: author PDF nn.2479.pdf attached to the EPFL Infoscience record) — https://infoscience.epfl.ch/entities/publication/58c0525d-2180-48a1-b48e-a2db6dddfce0

## [92] — 1.0

**Urbanczik & Senn (2014), "Learning by the dendritic prediction of somatic spiking", Neuron 81(3), 521–528, doi:10.1016/j.neuron.2013.11.030**

Read this one. Urbanczik and Senn split a neuron into a dendrite and a soma and ask the dendrite to predict what the soma will do; the synapses on the dendrite then change in proportion to the presynaptic input times the difference between what the soma actually did and what the dendritic potential predicted it would do. That difference — actual spike minus predicted firing probability — is the whole rule, and it is the same object walnutbutter calls the centred Hebbian eligibility: x times (n minus n-bar), or in the single-spike form, y minus p-hat. Where you get the prediction from a running count, they get it from a second compartment, but the algebra and the reason it works are the same, and their version is explicitly non-Hebbian in the sense that it learns nothing when the prediction is already right. The second half of the paper is the part that makes it must-read for you rather than merely interesting: they show the identical rule becomes unsupervised, supervised, or reinforcement learning depending only on what drives the soma and whether a reward signal scales the learning rate — a single three-factor rule covering all three paradigms, which is precisely the claim your own architecture is making. The preprint cites it only as prior event-driven NEST plumbing, which badly undersells it; it is the direct scholarly ancestor of the eligibility term you have already implemented, and it is short, readable, and free on the publisher's site.

*Cited in the preprint for:* Cited alongside [91] as the second voltage-based plasticity model already re-implemented event-driven in NEST (UrbanczikArchivingNode), used as the design precedent for their own archiving classes.

*Readable:* open access (Cell Press open archive; the site blocks automated fetching, so I confirmed the free status through Unpaywall and OpenAlex rather than by loading the PDF) — https://www.cell.com/neuron/fulltext/S0896-6273(13)01127-6

## [93] — 0.8

**Martín-Sánchez, Bohté & Otte (2022), "A Taxonomy of Recurrent Learning Rules", arXiv:2207.11439, doi:10.48550/arxiv.2207.11439**

A short, careful derivation paper — note it is a duplicate of entry [87] in the same bibliography. The authors start from backpropagation through time, derive real-time recurrent learning out of it step by step in one consistent notation, and then show exactly where e-prop sits: e-prop is RTRL with the cross-neuron terms of the sensitivity tensor thrown away, keeping only each synapse's effect on its own postsynaptic neuron. Having made that precise they generalize, producing a family of learning rules of which e-prop is one member, indexed by how much of the recurrent dependency you are willing to carry. This is the cleanest single document for understanding what an eligibility trace is approximating and what it is discarding, and the preprint leans on it twice in its appendix on online algorithms — once to classify e-prop and RTRL as causal forward-pass algorithms, and once to make the point that updating weights continuously, as they do, gives results distinct from BPTT, RTRL and e-prop alike. For you it is one step removed rather than central: walnutbutter's rule is a REINFORCE score function, not a truncated gradient, so nothing here is your machinery. But your "hazard" eligibility is a per-synapse quantity carried forward in time, and this paper tells you precisely what class of object that is and what the gradient-approximation people gave up to get one. It is twelve pages and free.

*Cited in the preprint for:* Cited twice in the appendix on online recurrent learning: to classify e-prop and RTRL as online/causal algorithms computable during the forward pass, and to note that continuous weight updating produces results distinct from BPTT, RTRL and e-prop.

*Readable:* preprint — https://arxiv.org/abs/2207.11439

## [94] — 0.3

**Budik & Elhanany (2006), "TRTRL: A localized resource-efficient learning algorithm for recurrent neural networks", 49th IEEE International Midwest Symposium on Circuits and Systems, vol. 1, 371–374, doi:10.1109/MWSCAS.2006.382075**

A four-page circuits-conference paper proposing Truncated RTRL. Real-time recurrent learning keeps, for every synapse, the sensitivity of every neuron in the network to that synapse — a three-index object costing storage that grows as the cube of the network size and work that grows as the fourth power, which is why nobody runs it. Budik and Elhanany simply delete most of it: each neuron keeps sensitivities only for the weights on its own incoming and outgoing links, which drops storage and computation to the square of the network size and, crucially for their purpose, makes each update depend only on quantities the neuron has locally, so the thing can be laid out in hardware. They report the accuracy loss as minor on small benchmarks. The preprint cites it in passing, as an example of an algorithm that updates weights partway through a sample rather than at the end. Its interest to you is genealogical and slight: this is an early, hardware-motivated instance of the exact move that later produced e-prop — throw away the cross-neuron part of the sensitivity tensor and what remains is local and affordable — made fifteen years earlier and without any of the biology. You are not doing RTRL and nothing here touches your hazard or your reward scalar. The IEEE version is paywalled; the author's master's thesis covering the same work is free.

*Cited in the preprint for:* Cited as an example of a truncated algorithm that updates weights midway through a sample rather than once per mini-batch.

*Readable:* paywalled (free version: Budik's 2006 University of Tennessee master's thesis, which develops the same TRTRL algorithm at length) — https://trace.tennessee.edu/utk_gradthes/1513/

## [95] — 0.6

**Kag & Saligrama (2021), "Training Recurrent Neural Networks via Forward Propagation Through Time", Proceedings of the 38th International Conference on Machine Learning, PMLR vol. 139, 5189–5200**

Kag and Saligrama propose an alternative to backpropagation through time that never unrolls the sequence at all. At each timestep they update the weights by minimizing an instantaneous objective — the current loss plus a regularizer that is itself dynamic, built from the losses already seen — and they show that following this sequence of cheap local problems converges to a stationary point of the real full-sequence objective. So the weights move at every single step, memory cost does not grow with sequence length, and the exploding and vanishing gradients that come from differentiating through hundreds of steps never arise; empirically plain LSTMs trained this way beat BPTT on long-dependency benchmarks. The preprint cites it as the reference case for updating weights at every step, the far end of the spectrum from once-per-mini-batch. That is why it matters to you rather than to them: walnutbutter runs forever and never stops to converge, its weights change continuously while the input is still arriving, and this paper is the ML community's principled account of why doing that can still be sound and what extra term you apparently need in the objective to make it so. It is exact-gradient work on ordinary recurrent nets, not spiking ones, so read it for the idea and not for anything you would implement; [96] is the spiking sequel.

*Cited in the preprint for:* Cited as the example of updating weights at every step (as opposed to midway through a sample or once per mini-batch) in the discussion of truncated and online algorithms.

*Readable:* proceedings (free) — https://proceedings.mlr.press/v139/kag21a.html

## [96] — 0.7

**Yin, Corradi & Bohté (2023), "Accurate online training of dynamical spiking neural networks through Forward Propagation Through Time", Nature Machine Intelligence 5(5), 518–527, doi:10.1038/s42256-023-00650-4**

This carries Kag and Saligrama's forward-propagation idea into spiking networks, and it is the most direct rival to e-prop in this bibliography. The authors argue that the online approximations to backpropagation through time — e-prop and OSTL among them — still cost a lot of memory and, being approximations, never quite match offline training. Their answer is to drop approximation entirely and use FPTT: minimize a dynamically regularized running risk, update at every timestep, memory cost fixed regardless of how long the sequence is. Paired with a neuron they introduce whose time constant is itself a learned dynamic variable (the liquid time-constant spiking neuron), FPTT-trained spiking networks beat the online approximations outright and match or exceed full offline BPTT on temporal classification, which is not a result anyone had before. For you the relevance is real but oblique. It is not your rule — it is still a gradient method with a surrogate, still needs a differentiable path, and your learning is a global scalar times a local score function. But it is the strongest published statement that a spiking network can be trained genuinely online with bounded memory and lose nothing, which is the claim your own architecture rests on, and it is the paper that would tell you what you are competing against if you ever want to argue walnutbutter's numbers. Free on arXiv.

*Cited in the preprint for:* Cited as the demonstration that per-step weight updating (FPTT, ref. 95) also works in spiking neural networks.

*Readable:* paywalled (free version: arXiv preprint, plus an author copy in the CWI institutional repository) — https://arxiv.org/abs/2112.11231

