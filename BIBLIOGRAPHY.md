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

2. Fiete, I. R., & Seung, H. S. (2006). Gradient learning in spiking neural
   networks by dynamic perturbation of conductances. *Physical Review
   Letters*, 97(4), 048104. https://doi.org/10.1103/PhysRevLett.97.048104
   Node perturbation for spiking neurons: perturb each neuron's input,
   correlate the perturbation with the global reward, and you have an
   unbiased estimate of the reward gradient without any backward pass. The
   `--late ignore` rule is this estimator proper; the exploration noise
   added to potentials each epoch is the perturbation.

## Local eligibility, global signal: three-factor rules

3. Izhikevich, E. M. (2007). Solving the distal reward problem through
   linkage of STDP and dopamine signaling. *Cerebral Cortex*, 17(10),
   2443-2452. https://doi.org/10.1093/cercor/bhl152
   Each synapse keeps a decaying eligibility trace written by pre/post spike
   timing; a later dopamine signal turns the trace into a weight change. The
   biological answer to "how does a synapse know it was on the trace that
   succeeded": it doesn't, the coincidence stands in for causality.

4. Legenstein, R., Pecevski, D., & Maass, W. (2008). A learning theory for
   reward-modulated spike-timing-dependent plasticity with application to
   biofeedback. *PLoS Computational Biology*, 4(10), e1000180.
   https://doi.org/10.1371/journal.pcbi.1000180
   What reward-modulated STDP can and cannot learn, and why the reward must
   correlate with the local eligibility for learning to happen. Relevant to
   the decoded critic's flat reward, and to the sign of the `--late depress`
   rule.

5. Frémaux, N., & Gerstner, W. (2016). Neuromodulated spike-timing-dependent
   plasticity, and theory of three-factor learning rules. *Frontiers in
   Neural Circuits*, 9, 85. https://doi.org/10.3389/fncir.2015.00085
   The review that names the family: presynaptic activity x postsynaptic
   activity (or perturbation) x a global third factor. Places node
   perturbation and reward-modulated Hebbian/STDP rules side by side, which
   is the choice `--late` exposes.

## The data

6. LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based
   learning applied to document recognition. *Proceedings of the IEEE*,
   86(11), 2278-2324. https://doi.org/10.1109/5.726791
   The MNIST digits: 60,000 training and 10,000 test images of 28 x 28
   bytes with their labels, as distributed by LeCun, Cortes and Burges. The
   mnist problem (AUTHORITY.md §8) reads the training split from `mnist/`,
   averaged to 14 x 14 and thresholded at half; the test split is not
   fetched, by decision; `walnutbutter.mnist` is the loader.

## Not yet mentioned in our conversations, but the roots of the above

- Markram, H., Lübke, J., Frotscher, M., & Sakmann, B. (1997). Regulation
  of synaptic efficacy by coincidence of postsynaptic APs and EPSPs.
  *Science*, 275(5297), 213-215. The first report that the order of pre and
  post spikes sets the sign of the change.
- Bi, G.-Q., & Poo, M.-M. (1998). Synaptic modifications in cultured
  hippocampal neurons: dependence on spike timing, synaptic strength, and
  postsynaptic cell type. *Journal of Neuroscience*, 18(24), 10464-10472.
  The STDP window itself: potentiation for pre-before-post, depression for
  post-before-pre, over tens of milliseconds.
- Werfel, J., Xie, X., & Seung, H. S. (2003). Learning curves for stochastic
  gradient descent in linear feedforward networks. *NIPS 16*; extended in
  *Neural Computation*, 17(12), 2699-2718 (2005). Compares weight
  perturbation, node perturbation and backpropagation: why node
  perturbation's variance grows with the number of neurons, which is what
  the 14-column runs felt -- and why composing gradient-following updates
  across units is free in expectation and paid for in variance (AUTHORITY.md
  §0.1, §8).
- Bridle, J. S. (1990). Probabilistic interpretation of feedforward
  classification network outputs, with relationships to statistical pattern
  recognition. In F. Fogelman Soulié & J. Hérault (Eds.), *Neurocomputing:
  Algorithms, Architectures and Applications* (NATO ASI Series F, Vol. 68,
  pp. 227-236). Springer. Names the softmax and pairs it with the
  log-likelihood of the true class: the evidence critic of §8 is this
  reading of the class sums at a temperature.
- Bishop, C. M. (1995). *Neural Networks for Pattern Recognition*. Oxford
  University Press. §6.9, cross-entropy for multiple classes: the gradient
  of the softmax cross-entropy with respect to its inputs is q_k - t_k,
  which is the evidence critic's push of (1 - q_y)/T on the label's
  population and cost of q_k/T on every other; and the softmax as a smooth
  winner-take-all, the class critic being its T -> 0 limit.
