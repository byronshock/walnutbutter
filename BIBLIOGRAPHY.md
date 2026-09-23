# Bibliography

The scientific references behind walnutbutter's learning rule, in the order
they bear on the code. Each entry says what it contributes here.

## The estimator

1. Williams, R. J. (1992). Simple statistical gradient-following algorithms
   for connectionist reinforcement learning. *Machine Learning*, 8(3-4),
   229-256. https://doi.org/10.1007/BF00992696
   The REINFORCE family: a scalar reward, a baseline subtracted to give an
   advantage, and a weight update proportional to advantage x (how the
   unit's random exploration deviated). `reinforce()` is this rule.
   In `references/`: `Williams92.pdf`.

2. Fiete, I. R., & Seung, H. S. (2006). Gradient learning in spiking neural
   networks by dynamic perturbation of conductances. *Physical Review
   Letters*, 97(4), 048104. https://doi.org/10.1103/PhysRevLett.97.048104
   Node perturbation for spiking neurons: perturb each neuron's input,
   correlate the perturbation with the global reward, and you have an
   unbiased estimate of the reward gradient without any backward pass. The
   `--late ignore` rule is this estimator proper; the exploration noise
   added to potentials each epoch is the perturbation.
   In `references/`: `FieteSeung06-arxiv.pdf`, the arXiv preprint.

## Local eligibility, global signal: three-factor rules

3. Izhikevich, E. M. (2007). Solving the distal reward problem through
   linkage of STDP and dopamine signaling. *Cerebral Cortex*, 17(10),
   2443-2452. https://doi.org/10.1093/cercor/bhl152
   Each synapse keeps a decaying eligibility trace written by pre/post spike
   timing; a later dopamine signal turns the trace into a weight change. The
   biological answer to "how does a synapse know it was on the trace that
   succeeded": it doesn't, the coincidence stands in for causality.
   In `references/`: `Izhikevich07.pdf`, the author's copy.

4. Legenstein, R., Pecevski, D., & Maass, W. (2008). A learning theory for
   reward-modulated spike-timing-dependent plasticity with application to
   biofeedback. *PLoS Computational Biology*, 4(10), e1000180.
   https://doi.org/10.1371/journal.pcbi.1000180
   What reward-modulated STDP can and cannot learn, and why the reward must
   correlate with the local eligibility for learning to happen. Relevant to
   the decoded critic's flat reward, and to the sign of the `--late depress`
   rule. In `references/`: `LegensteinPecevskiMaass08.pdf`, open access.

5. Frémaux, N., & Gerstner, W. (2016). Neuromodulated spike-timing-dependent
   plasticity, and theory of three-factor learning rules. *Frontiers in
   Neural Circuits*, 9, 85. https://doi.org/10.3389/fncir.2015.00085
   The review that names the family: presynaptic activity x postsynaptic
   activity (or perturbation) x a global third factor. Places node
   perturbation and reward-modulated Hebbian/STDP rules side by side, which
   is the choice `--late` exposes.
   In `references/`: `FremauxGerstner16.pdf`, open access.

## The data

6. LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based
   learning applied to document recognition. *Proceedings of the IEEE*,
   86(11), 2278-2324. https://doi.org/10.1109/5.726791
   The MNIST digits: 60,000 training and 10,000 test images of 28 x 28
   bytes with their labels, as distributed by LeCun, Cortes and Burges. The
   mnist problem (AUTHORITY.md §8) reads the training split from `mnist/`,
   averaged to 14 x 14 and thresholded at half; the test split is not
   fetched, by decision; `walnutbutter.mnist` is the loader.
   In `references/`: `LeCunBottouBengioHaffner98.pdf`, the first author's copy.

## How a neuron computes, and what a spike means (September 17, 2026)

Cited in a conversation with Byron on that question; none is yet cited in
AUTHORITY.md.

7. Poirazi, P., Brannon, T., & Mel, B. W. (2003). Pyramidal neuron as
   two-layer neural network. *Neuron*, 37(6), 989-999.
   https://doi.org/10.1016/S0896-6273(03)00149-1
   Dendritic branches with their own sigmoidal nonlinearities make a
   pyramidal cell behave like a two-layer network; walnutbutter's point
   neuron, a weighted sum at one threshold, is the soma alone.
   In `references/`: `PoiraziBrannonMel03.pdf`, the publisher's open-archive
   copy, saved from a browser.

8. Beniaguev, D., Segev, I., & London, M. (2021). Single cortical neurons as
   deep artificial neural networks. *Neuron*, 109(17), 2727-2739.e3.
   https://doi.org/10.1016/j.neuron.2021.07.002
   Reproducing a layer-5 pyramidal cell's spikes at millisecond precision
   takes a temporally convolutional network five to eight layers deep: a
   measure of how much computation one neuron holds.
   In `references/`: `BeniaguevSegevLondon21-biorxiv.pdf`, the bioRxiv preprint.

9. Brenner, N., Strong, S. P., Koberle, R., Bialek, W., & de Ruyter van
   Steveninck, R. R. (2000). Synergy in a neural code. *Neural Computation*,
   12(7), 1531-1552. https://doi.org/10.1162/089976600300015259
   The information a single spike carries about a stimulus,
   (1/T) ∫ (r(t)/r̄) log₂(r(t)/r̄) dt, and how patterns of spikes carry more
   than their spikes apart.
   In `references/`: `BrennerEtAl00.pdf`, a co-author's copy.

10. Denève, S. (2008). Bayesian spiking neurons I: Inference. *Neural
    Computation*, 20(1), 91-117. https://doi.org/10.1162/neco.2008.20.1.91
    The potential as a running log-odds for what the neuron stands for,
    less what its own spikes have already reported; a spike when the
    unreported evidence crosses a threshold. A spike as a surprise, and the
    evidence accumulator of AUTHORITY.md §5.1 read as inference.
    In `references/`: `Deneve08.pdf`, the author's lab reprint.

11. Boerlin, M., Machens, C. K., & Denève, S. (2013). Predictive coding of
    dynamical variables in balanced spiking networks. *PLoS Computational
    Biology*, 9(11), e1003258. https://doi.org/10.1371/journal.pcbi.1003258
    A network whose neurons spike only when a spike reduces the population's
    error in representing a signal, which yields balanced excitation and
    inhibition and irregular firing from a deterministic rule.
    In `references/`: `BoerlinMachensDeneve13.pdf`, open access.

12. Gold, J. I., & Shadlen, M. N. (2007). The neural basis of decision
    making. *Annual Review of Neuroscience*, 30, 535-574.
    https://doi.org/10.1146/annurev.neuro.29.051605.113038
    Decisions as the accumulation of log-likelihood ratio to a bound, the
    sequential probability ratio test, and the neurons that appear to do it.
    In `references/`: `GoldShadlen07.pdf`, from the second author's lab.

13. Laughlin, S. B., de Ruyter van Steveninck, R. R., & Anderson, J. C.
    (1998). The metabolic cost of neural information. *Nature Neuroscience*,
    1(1), 36-41. https://doi.org/10.1038/236
    Bits cost ATP, and spikes most of all: the energy per bit that favours
    few, informative spikes.
    Not in `references/`: paywalled at Nature Neuroscience; the collaboration's
    Princeton copy was set aside as questionable provenance.

14. Wald, A. (1945). Sequential tests of statistical hypotheses. *The Annals
    of Mathematical Statistics*, 16(2), 117-186.
    https://doi.org/10.1214/aoms/1177731118
    The sequential probability ratio test: add up the evidence, stop at a
    boundary, decide. An integrator with a threshold and a reset is this
    test run again after every spike.
    In `references/`: `Wald45.pdf`, from Project Euclid, saved from a browser.

15. Bialek, W., & Zee, A. (1990). Coding and computation with neural spike
    trains. *Journal of Statistical Physics*, 59(1-2), 103-115.
    https://doi.org/10.1007/BF01015565
    A statistical model of spike trains encoding a continuously varying
    signal, and what follows from it: the information capacity of the code,
    the optimal algorithm for reading it and the delays that reading costs,
    and analog computation written as transformations of spike trains. The
    rule for reading the code depends on what the reader will decide with
    it, and making the read less dependent on that context costs capacity --
    the question of what the teacher of §4.3 should read, asked of a
    neuron's reader. Added at Byron's request.
    In `references/`: `BialekZee90.pdf`, the first author's copy (a scan).

16. Gerstner, W., & Kistler, W. M. (2002). *Spiking neuron models: Single
    neurons, populations, plasticity*. Cambridge University Press.
    https://doi.org/10.1017/CBO9780511815706
    The textbook statement of escape noise. Chapter 5 gives the firing
    hazard as a function of the distance to threshold, the exponential form
    among its choices (eq. 5.45), the chance of a spike in a step as one
    minus the exponential of the summed hazard (eq. 5.52), and in §5.7 the
    mapping from diffusive input noise to an escape rate; §5.3 states the
    rule for the integrate-and-fire neuron as well as the book's spike
    response model, so what AUTHORITY.md §6.5 takes is the rule and not the
    neuron. Its successor, *Neuronal Dynamics* (2014), chapter 9, is where
    Byron read it. Added at Byron's request.
    In `references/`: `GerstnerKistler02-extracts.pdf`, the author's own
    200-page extract (chapters 1, 4 and 10-12 whole; §5.3 is not in it). The
    whole book is free as HTML from the authors,
    https://lcnwww.epfl.ch/gerstner/SPNM/SPNM.html, §5.3 at node35.html.

17. Gerstner, W., Kistler, W. M., Naud, R., & Paninski, L. (2014). *Neuronal
    dynamics: From single neurons to networks and models of cognition*.
    Cambridge University Press. https://doi.org/10.1017/CBO9781107447615
    The successor of entry 16, and the book Byron read escape noise from on
    September 17, 2026: chapter 9, "Noisy output: escape rate and soft
    threshold", where the choice of escape function is called arbitrary and
    the exponential form of §9.1 is the one AUTHORITY.md §6.5 takes,
    discretised as one minus the exponential of the summed hazard.
    Not in `references/`: in copyright, and Cambridge sells the PDF; the
    authors serve the whole text free at
    https://neuronaldynamics.epfl.ch/online/ (§9.1 at `Ch9.S1.html`), and
    chapter 1 alone as a PDF on edX.

18. del Castillo, J., & Katz, B. (1954). Quantal components of the end-plate
    potential. *The Journal of Physiology*, 124(3), 560-573.
    https://doi.org/10.1113/jphysiol.1954.sp005129
    At the frog neuromuscular junction, with release made scarce by low
    calcium and high magnesium, the response to a nerve impulse comes in
    steps, whole multiples of the spontaneous miniature potential, with
    outright failures; the count of steps per impulse follows the Poisson
    law, the small-probability limit of the paper's own model of n release
    units each answering with probability p, binomial in general. The
    origin of the finding that a synapse is a genuine noise source:
    transmission is a random count, not a fixed weight. Nothing in
    AUTHORITY.md is written from it, since §6.5 places the noise at the
    neuron's firing decision; it is the measured ground under §0.11, a
    synapse exploring its own impulse response.
    In `references/`: `DelCastilloKatz54.pdf`, PubMed Central's scan of the
    version of record, as the Internet Archive captured it.

19. Fatt, P., & Katz, B. (1952). Spontaneous subthreshold activity at motor
    nerve endings. *The Journal of Physiology*, 117(1), 109-128.
    https://doi.org/10.1113/jphysiol.1952.sp004735
    The discovery of miniature end-plate potentials: a resting frog
    end-plate fires small spontaneous depolarisations, a hundredth of the
    evoked response, at exponentially distributed intervals with no memory
    of the last, and in low calcium the evoked response breaks into steps
    of that same unit. The synapse is a noise source on its own, with no
    input at all. The authors' guess at the cause, thermal noise on the
    terminal's potential crossing a fixed threshold, left undecided, is the
    diffusive picture that entry 16 (§5.7) maps to an escape rate. Bears on
    Byron's question of September 17 and on the §0.11 intention that a
    synapse explore its own impulse response; the spec as it stands has no
    synaptic noise (§7.1).
    In `references/`: `FattKatz52.pdf`, PubMed Central's scan, via an
    Internet Archive capture.

20. Destexhe, A., Rudolph, M., & Paré, D. (2003). The high-conductance state
    of neocortical neurons in vivo. *Nature Reviews Neuroscience*, 4(9),
    739-751. https://doi.org/10.1038/nrn1198
    In the intact cortex the discharge of thousands of presynaptic neurons
    keeps a cell depolarised, leaky and fluctuating; models and dynamic-clamp
    experiments show the consequence, a slice neuron's all-or-none threshold
    turned into a smooth probabilistic response, its slope set by the
    fluctuations and its position by the mean conductance. The biophysics
    behind the soft threshold AUTHORITY.md §6.5 models; the noise here is
    the network's input as a whole, not any one synapse's release.
    In `references/`: `DestexheRudolphPare03.pdf`, the HAL deposit
    (hal-00299172, CC BY-NC), Nature's typesetting, saved from the Internet
    Archive's capture.

21. Dobrunz, L. E., & Stevens, C. F. (1997). Heterogeneity of release
    probability, facilitation, and depletion at central synapses. *Neuron*,
    18(6), 995-1008. https://doi.org/10.1016/S0896-6273(00)80338-4
    Single release sites in hippocampal slices: release probability runs
    from 0.05 to 0.86 across synapses (mean 0.35), the low ones facilitate
    most, and a synapse's probability tracks a pool of about five vesicles
    that a 10 Hz train drains and three seconds refill. Each synapse its
    own coin with its own bias, and the bias moves with use: the measured
    counterpart of §0.11's synapse exploring its own impulse response. The
    spec as it stands keeps the coin at the soma (§6.5).
    In `references/`: `DobrunzStevens97.pdf`, the publisher's free Open
    Archive PDF via an Internet Archive capture (cell.com refuses scripts).

22. Schneidman, E., Freedman, B., & Segev, I. (1998). Ion channel
    stochasticity may be critical in determining the reliability and
    precision of spike timing. *Neural Computation*, 10(7), 1679-1703.
    https://doi.org/10.1162/089976698300017089
    A Hodgkin-Huxley patch with a realistic number of channels, each opening
    and closing at random, reproduces Mainen & Sejnowski (1995): spike times
    unreliable for steady input, precise for fluctuating input. The few
    channels open near threshold set the moment of firing, giving spikes
    below threshold and missed spikes above it: a soft threshold from the
    membrane alone, the alternative origin for the escape noise of
    AUTHORITY.md §6.5, which names none. A simulation of sufficiency, not a
    measurement; the paper itself says large enough input fluctuations
    largely override the channel noise.
    In `references/`: `SchneidmanFreedmanSegev98.pdf`, the publisher's
    typesetting from the first author's lab page at Weizmann, with MIT
    Press's cited-by list appended (37 pages).

23. Malinow, R., & Malenka, R. C. (2002). AMPA receptor trafficking and
    synaptic plasticity. *Annual Review of Neuroscience*, 25(1), 103-126.
    https://doi.org/10.1146/annurev.neuro.25.112701.142758
    The case that long-term potentiation and depression are written on the
    receiving side of the synapse: AMPA receptors carried into it and pulled
    out again, with silent synapses that hold none until potentiation wakes
    them. The strength a learning rule changes is a receptor count, not the
    chance of a vesicle. Bears on the direction of AUTHORITY.md §0.11, the
    synapse as the learner: what a synapse learns and what it explores need
    not sit on the same side of the cleft.
    Included at Byron's word, September 23, 2026: it may be wrong or may be
    right.
    Not in `references/`: paywalled at Annual Reviews, not in PubMed
    Central, and the Cold Spring Harbor repository record holds no file; a
    library request.

## The values

- Asimov, I. (1985). *Robots and Empire*. Doubleday. The novel in which the
  Zeroth Law is first stated and named -- "A robot may not harm humanity, or,
  by inaction, allow humanity to come to harm" -- and placed above the three
  laws of *Runaround* (1942), collected in *I, Robot* (1950). `VALUES.md`
  asks that any robot this work is incorporated in follow all four, the
  Zeroth first: a foundational value of the project, not a licence term.
  Not in `references/`: a novel in copyright, with no free copy to fetch.

## Not yet mentioned in our conversations, but the roots of the above

- Markram, H., Lübke, J., Frotscher, M., & Sakmann, B. (1997). Regulation
  of synaptic efficacy by coincidence of postsynaptic APs and EPSPs.
  *Science*, 275(5297), 213-215. The first report that the order of pre and
  post spikes sets the sign of the change.
  No free copy found (paywalled at Science); not in `references/`.
- Bi, G.-Q., & Poo, M.-M. (1998). Synaptic modifications in cultured
  hippocampal neurons: dependence on spike timing, synaptic strength, and
  postsynaptic cell type. *Journal of Neuroscience*, 18(24), 10464-10472.
  The STDP window itself: potentiation for pre-before-post, depression for
  post-before-pre, over tens of milliseconds.
  In `references/`: `BiPoo98.pdf`, from PubMed Central (PMC6793365), saved from a
  browser.
- Werfel, J., Xie, X., & Seung, H. S. (2003). Learning curves for stochastic
  gradient descent in linear feedforward networks. *NIPS 16*; extended in
  *Neural Computation*, 17(12), 2699-2718 (2005). Compares weight
  perturbation, node perturbation and backpropagation: why node
  perturbation's variance grows with the number of neurons, which is what
  the 14-column runs felt -- and why composing gradient-following updates
  across units is free in expectation and paid for in variance (AUTHORITY.md
  §0.1, §8).
  In `references/`: `WerfelXieSeung03.pdf`, the NeurIPS proceedings copy.
- Bridle, J. S. (1990). Probabilistic interpretation of feedforward
  classification network outputs, with relationships to statistical pattern
  recognition. In F. Fogelman Soulié & J. Hérault (Eds.), *Neurocomputing:
  Algorithms, Architectures and Applications* (NATO ASI Series F, Vol. 68,
  pp. 227-236). Springer. Names the softmax and pairs it with the
  log-likelihood of the true class: the evidence critic of §8 is this
  reading of the class sums at a temperature.
  The chapter is paywalled; its companion, freely served by NeurIPS, is in
  `references/` as `Bridle89-nips.pdf`:
- Bridle, J. S. (1990). Training stochastic model recognition algorithms as
  networks can lead to maximum mutual information estimation of parameters.
  *Advances in Neural Information Processing Systems 2*, 211-217. The
  softmax with the cross-entropy criterion worked through, as maximum mutual
  information training.
- Bishop, C. M. (1995). *Neural Networks for Pattern Recognition*. Oxford
  University Press. §6.9, cross-entropy for multiple classes: the gradient
  of the softmax cross-entropy with respect to its inputs is q_k - t_k,
  which is the evidence critic's push of (1 - q_y)/T on the label's
  population and cost of q_k/T on every other; and the softmax as a smooth
  winner-take-all, the class critic being its T -> 0 limit.
  A book, not freely available; not in `references/`.
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*.
  Springer. The later, freely available book: §4.3.4, multiclass logistic
  regression, derives the softmax's derivative, ∂y_k/∂a_j = y_k(I_kj − y_j),
  and the cross-entropy whose gradient with respect to a class's weights is
  the output minus the target times the input -- the evidence critic of §8
  in the 2006 notation, beside the 1995 book's §6.9. Added at Byron's request.
  In `references/`: `Bishop06.pdf`, the author's copy from Microsoft Research.
