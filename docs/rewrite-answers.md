# The rewrite: what Byron has decided, in order

*Answers to `rewrite-decisions.md`, in Byron's words, as they are given. A
clause is written into `AUTHORITY.md` only from an answer here.*

## 1. What fires a neuron, once θ is deprecated?

**Answered September 17, 2026, 14:50 MDT.** *"We will eventually get to A!
This is exciting. But first we must tackle neural spike escape noise as a
potential mechanism. I will argue later that it's not the mechanism, but we
have a learning rule derived for it right now that is working right now, at
least in a feedforward fashion, and not well."*

So: **escape noise fires the neuron, for now** — the firing decision is a
draw whose rate rises with the potential (option B of the three offered) —
and the eligibility derived from it, the single-spike rule's hazard row,
survives with it. Option A, exploration alone with the potential driving
nothing, is where the project is going, not where the rewrite starts.

*What this leaves open, and is asked next:* the escape hazard's rate is
quoted today as a margin above θ in units of ESCAPE_DELTA × θ, so with θ
deprecated the rule needs a scale from somewhere else.
