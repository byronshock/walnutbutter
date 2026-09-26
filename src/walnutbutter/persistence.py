"""Saving and restoring what the network has learned.

A checkpoint is a JSON file holding every connection's weight (by ID) together
with what is needed to rebuild the identical network: count, the wiring
rule, threshold and seed. Loading rebuilds it from
those settings
and copies the weights back in, so a run that took hours can be continued or
inspected later.

Under exploration at the synapse (AUTHORITY.md §7.5) a checkpoint also
carries each neuron's gain, each output's read count this epoch, the ventured
mark on every signal in flight and the settings of §7.1, §7.6, §7.7, §8.17 and
5.4b (§8.14, §12.9), and is written as format 3, which a reader that knows
only format 2 refuses rather than resume as a run of the neuron rule. A
neuron-rule checkpoint stays format 2, readable by that reader as before, and
names its exploration too; one that names none -- every file written before
September 25, 2026 -- was saved under the neuron rule (§8.14).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .constants import GOO_SCALING_FACTOR
from .goo import Goo, scaled_projection
from .network import EXPLORATIONS, TRACE_VENTURED_BIAS, Network
from .neuron import Neuron

FORMAT = 2  # 1 held a permutation, an input coding, an error-correcting code and a flip; §5.2 dropped all four
SYNAPSE_FORMAT = 3  # a checkpoint saved under exploration at the synapse: format 2 and what §8.14 adds for it. A reader
# of format 2 alone refuses it, as it must: it would resume the run under the neuron rule (§12.9)
FORMATS = (FORMAT, SYNAPSE_FORMAT)  # what read_checkpoint accepts, so every format-2 file on disk still loads


def checkpoint(network: Network, path: str | Path, teacher=None) -> dict:
    """Write the network's weights and settings to `path`. Returns what was written.

    Everything a resume needs to be the same run continued (§12.9, §12.11),
    the signals in flight among it and, beside them, the stimuli and charges
    still waiting in the queue -- an arrival drawn within the clock's slack of
    the horizon waits for the next epoch (§3.4). Under exploration at the
    synapse the file is format 3 and carries the gains, the read counts, the
    ventured marks and the settings (§8.14); a network given the charged drive
    under the neuron rule, which no run may use (5.4b), is refused rather than
    written as one that a reader would run.
    """
    engine = getattr(network, "engine", "objects")
    if engine == "arrays":
        network.sync_to_mesh()  # the checkpoint is written from the mesh, whichever engine ran it
        network = network.mesh
    synaptic = getattr(network, "exploration", "neuron") == "synapse"
    if getattr(network, "drive", None) == "charged" and not synaptic:
        raise ValueError("the charged drive runs under exploration at the synapse only (5.4b): this network explores by "
                         "the neuron rule, and a checkpoint of it would be a run the file refuses (§12.2)")
    data = {
        "format": SYNAPSE_FORMAT if synaptic else FORMAT,
        "exploration": "synapse" if synaptic else "neuron",  # §7.1, whichever it is (§8.14)
        "container": "goo",
        "across": network.across,
        "rows": network.rows,
        "threshold": network.threshold,
        "minimum_potential": network.minimum_potential,
        "seed": network.seed,
        "weight": getattr(network, "weight", None),
        "weight_range": list(network.weight_range),
        "population": network.population,  # neurons per raw bit under population coding
        "temperature": network.temperature,  # the evidence critic's temperature (§8)
        "clock": network.clock,  # clock neurons at the front of the input zone, always driven (§4.3)
        "quash": [network.quash_rate, network.quash_k],  # the cycle quash (§6.11)
        "drive": network.drive,  # how a bit becomes spikes (§4.3)
        "rate": [network.rate_on, Neuron.rate_tau],  # the rate read: saturation in Hz, and its window in ms (§4.3)
        "pickiness": network.pickiness,  # the count read's line in spikes (§5.10, §9.5)
        "input_rate": [network.input_rate, network.input_rate_off],  # per ms, under rate drive
        "problem": getattr(network, "problem", None),  # what the run was asked to do (problems.PROBLEMS)
        "engine": engine,
        "random_weights": network.weight is None,
        "epoch": network.epoch,
        "time": network.time,  # the clock, nominal milliseconds
        "interval": network.interval,
        "presentation_time": network.presentation,  # §5.4a; None is the whole epoch, which is the default
        "tau": Neuron.tau,
        "refractory": Neuron.refractory,
        "hop": Neuron.hop,
        "bored_after": Neuron.bored_after,
        "horizon": network.horizon,  # the time the schedule has run to
        "connections": len(network.connections),
        "wiring_digest": wiring_digest(network),  # §12.11: proof the rebuild is edge for edge the same mesh
        # §12.11: each generator's state, not its seed and a count of draws, so the next number is the number the
        # uninterrupted run would have taken. The network's own stream has drawn the wiring and the weights and goes
        # on drawing the drive's arrival times; the exploration stream supplies the firing decisions (§7.3).
        "network_state": _stream_state(getattr(network, "_rng", None)),
        "explore_state": _stream_state(getattr(network, "explore_rng", None)),
        "input_at": getattr(network, "input_at", 0),  # how far through a pre-drawn input stream the run has got
        "weights": [network.connections[i].weight for i in range(1, len(network.connections) + 1)],
        "thresholds": [n.threshold for n in network.all_neurons()],
        # per neuron since §5.2: a container that scales with fan-in gives each its own floor, not the one scalar
        "floors": [n.minimum_potential for n in network.all_neurons()],
        "rates": [n.rate for n in network.all_neurons()],
        # the single-spike rule (§6.7, September 17, 2026): p_hat_j, the decisions to date, and E_j, the spikes expected since the
        # neuron's last spike, so a resumed run charges from where it was
        "expectations": [n.expectation for n in network.all_neurons()],
        "decisions": [n.decisions for n in network.all_neurons()],
        "expected_since_spike": [n.expected for n in network.all_neurons()],
        # the state the clock leaves behind, so a resumed run continues rather than restarts
        "potentials": [n.potential for n in network.all_neurons()],
        "fired_at": [n.fired_at for n in network.all_neurons()],
        "last_update": [n.last_update for n in network.all_neurons()],
        "previous_fired_at": [n.previous_fired_at for n in network.all_neurons()],
        "spikes": [n.spikes for n in network.all_neurons()],
        # the synapses' stamps and the signals in flight, so a resumed run continues mid-cascade
        "last_signal": [network.connections[i].last_signal for i in range(1, len(network.connections) + 1)],
        # escape noise (§5.2): the width in units of the starting threshold, each neuron's own in potential units,
        # since when each hazard has run, and each synapse's trace of what it still has in its target (§6.7)
        "escape_delta": network.escape_delta,
        "deltas": [n.delta for n in network.all_neurons()],
        "exposed_since": [n.exposed_since for n in network.all_neurons()],
        "traces": [[network.connections[i].trace, network.connections[i].trace_at] for i in range(1, len(network.connections) + 1)],
        "notes": [network.connections[i].noted for i in range(1, len(network.connections) + 1)],  # B_ij, each open arrival's note (§6.7)
        # the signals in flight by connection id; under exploration at the synapse each with its ventured mark (§7.9)
        "pending": ([[time, connection.id, ventured] for time, connection, ventured in network.schedule.pending(marks=True)]
                    if synaptic else [[time, connection.id] for time, connection in network.schedule.pending()]),
        "waiting": _waiting(network),  # the stimuli and charges the queue still holds, by neuron (§3.4, 5.4b)
        "rule": network.rule,  # which rule pays at the read: "reinforce", or "local" for none (§9.1)
    }
    if synaptic:  # §8.14, §12.9: what exploration at the synapse adds; kappa_i is a rule and is recomputed (§7.7, §12.10)
        data.update({
            "synapse_hazard_rest": network.synapse_hazard_rest,  # h0 (§7.6)
            "synapse_hazard_family": network.synapse_hazard_family,  # loglinear or linear (§7.6)
            "synapse_hazard_scaling": network.synapse_hazard_scaling,  # count or fan-out (§7.7)
            "trace_mode": network.trace_mode,  # all or ventured (§8.17)
            "drive_steps": network.drive_steps,  # 5.4b's DRIVE_STEPS, a run option carried whatever the drive
            "gains": [n.gain for n in network.all_neurons()],  # G_j, open across the reads (§8.16)
            "read_counts": [n.read_count for n in network.all_neurons()],  # each output's read synapse this epoch (§7.9)
        })
        if network.trace_mode == "ventured":
            data["estimator_bias"] = TRACE_VENTURED_BIAS  # the run says so in its record (§8.17)
    data["count"] = network.count  # goo's whole topology: no positions to record and nothing drawn to verify
    data["scale_with_fan_in"] = network.scale_with_fan_in_on  # whether §5.2's rescaling built those floors
    data["projection"] = network.projection  # the three earlier wirings' probability (§3.4)
    data["scaling_factor"] = network.scaling_factor  # the scaled rule's knob (§3.4): N times it heard in expectation
    data["outputs"] = network.outputs  # the output zone's width, when it differs from the input's (§8, mnist)
    data["wiring"] = network.wiring  # the rule (§3.4), or one of the three before it; a checkpoint restores under its own
    if teacher is not None:
        data["learning"] = {
            "rule": teacher.rule,
            "target": teacher.target,
            "lr": teacher.lr,
            "eligibility": teacher.eligibility,
            "epochs": teacher.epochs,
            "homeostasis": teacher.homeostasis,
            "target_rate": teacher.target_rate,
            "unstick": teacher.unstick,
            "unstick_target": teacher.unstick_target,
            "critic": teacher.critic,
            "history": teacher.history,
            "total_reward": teacher.total_reward,
            "baseline": teacher.baseline,
            "average": teacher.average,
        }
    tmp = Path(path).with_suffix(Path(path).suffix + ".tmp")
    tmp.write_text(json.dumps(data))
    tmp.replace(path)  # atomic: a crash mid-write never leaves a half checkpoint
    return data


def _waiting(network) -> list[list]:
    """Every event in the queue that is not a signal, in the order the queue would take it, by neuron index: a stimulus
    as [time, "stimulus", i], a charge of theta / steps (5.4b) as [time, "charge", i, steps], an external input of a
    given amount as [time, "external", i, amount]. The drive draws an epoch's arrivals up to the horizon and the
    schedule runs what falls before it by more than the clock's slack (§3.4), so one drawn within that slack waits for
    the next epoch, and a checkpoint that dropped it would resume as another run (§12.11)."""
    from .propagation import SIGNAL, STIMULUS
    index = {id(neuron): i for i, neuron in enumerate(network.all_neurons())}
    waiting = []
    for time, kind, _, payload in sorted(network.schedule._heap, key=lambda e: e[:3]):
        if kind == SIGNAL:
            continue
        if kind == STIMULUS:
            waiting.append([time, "stimulus", index[id(payload)]])
        else:
            neuron, amount, steps = payload
            waiting.append([time, "charge", index[id(neuron)], steps] if amount is None
                           else [time, "external", index[id(neuron)], amount])
    return waiting


def across_of(data: dict) -> int:
    """The count across a checkpoint's input zone: "across", or "columns" in files written before the rename."""
    return data["across"] if "across" in data else data["columns"]


def _stream_state(rng) -> list | None:
    """A `random.Random`'s Mersenne state as a list of ints, or None where there is no stream.

    `getstate()` is `(3, tuple_of_625, None)`; only the middle matters, the
    version being fixed and the Gaussian spare unused by any draw this project
    takes. §12.11 wants the state and not the seed and a count of draws.
    """
    return None if rng is None else list(rng.getstate()[1])


def _restore_stream(rng, state) -> None:
    """Put a stream back where the checkpoint left it, if both the stream and the state are there."""
    if rng is not None and state:
        rng.setstate((3, tuple(int(x) for x in state), None))


def wiring_digest(network) -> str:
    """A digest of every synapse: the (source, target) index pairs in connection-id order, and the neuron count.

    §12.11 wants a resumed network "checked neuron for neuron and synapse for
    synapse against a fresh build from the seed, and refused if they differ".
    Comparing counts alone lets a checkpoint whose wiring differs edge for edge
    load silently, which is D3 of `docs/conformance-checks.md`. A digest checks
    every edge without putting a second copy of the topology in the file -- the
    mesh is rebuilt from the seed either way, so what is needed is proof the
    rebuild matches, not the edges themselves.
    """
    order = {id(neuron): i for i, neuron in enumerate(network.all_neurons())}
    h = hashlib.sha256(f"{len(order)} neurons".encode())
    for connection_id in range(1, len(network.connections) + 1):
        c = network.connections[connection_id]
        h.update(f"{order[id(c.source)]}>{order[id(c.target)]}:{c.kind};".encode())
    return h.hexdigest()


def hop_of(data: dict) -> float:
    """A checkpoint's hop in milliseconds: "hop", or converted from a file written before HOP replaced REFRACTORY_HOPS.

    The old field is the refractory period divided by the hop, so the hop is the
    quotient -- which preserves the timing the run actually used rather than
    imposing today's. The field records what a run ran; refusing it under §12.2
    would orphan every checkpoint already on disk, which is why this converts
    (AUTHORITY.md §3.2, and A1/A2 of `docs/conformance-checks.md`, as Byron
    settled it September 19, 2026).
    """
    if "hop" in data:
        return float(data["hop"])
    return float(data["refractory"]) / float(data["refractory_hops"])


def read_checkpoint(path: str | Path) -> dict:
    """A checkpoint's contents: format 2, or format 3 under exploration at the synapse; anything else is refused."""
    data = json.loads(Path(path).read_text())
    if data.get("format") not in FORMATS:
        # a format-1 file may carry a non-identity permutation, and §5.2 has no permutation: a run scored against a
        # scrambled zone would resume against an unscrambled one with no error at all, so refuse it by name
        scrambled = data.get("permutation")
        if scrambled is not None and list(scrambled) != sorted(scrambled):
            raise ValueError(f"{path}: written under a permuted input zone, which §5.2 dropped -- its weights were scored against a scrambling this build cannot reproduce")
        raise ValueError(f"{path}: unknown checkpoint format {data.get('format')!r}")
    return data


def restore(path: str | Path) -> tuple[Goo, dict]:
    """Rebuild the network described by the checkpoint and load its weights into it.

    Returns the network and the checkpoint data (so a Teacher can be resumed).
    """
    data = read_checkpoint(path)
    if data.get("container") != "goo":
        raise ValueError(
            f"{path}: container {data.get('container')!r} left the specification on September 17, 2026"
            " (AUTHORITY.md §4.1); it is readable at the tag lab-notebook-2026-09-17"
        )
    return _restore_goo(data), data


def _restore_goo(data: dict) -> Goo:
    """Rebuild goo from its count and its wiring rule, then load its weights.

    Where nothing about the topology was drawn -- the earlier wirings at
    projection 1, the scaled rule where every probability is 0 or 1 -- a goo
    built without a seed restores exactly; otherwise the seed decided the
    wiring and is needed. A checkpoint from
    before the zone rule (one that records `direct_projection`) cannot be
    rebuilt: its zones projected onto each other.
    """
    if "direct_projection" in data:
        raise ValueError(f"{path_of(data)}: goo built before the zone rule of September 14, 2026; its wiring cannot be rebuilt")
    # the wiring it was built under: recorded since the uniform rule of September 15; before that, the zone rule, with or
    # without the afternoon's equal fan-in (§3.4)
    wiring = data.get("wiring", "zones-equal" if data.get("equal_fan_in") else "zones")
    scaling_factor = data.get("scaling_factor", GOO_SCALING_FACTOR)
    if wiring in ("scaled", "scaled-open"):
        outputs = data.get("outputs") or across_of(data)
        drawn = any(0.0 < scaled_projection(data["count"], across_of(data), outputs, scaling_factor, zone, wiring == "scaled-open") < 1.0
                    for zone in ("input", "hidden", "output"))
        how = f"at scaling factor {scaling_factor:g}"
    else:
        drawn = data.get("projection", 1.0) < 1.0
        how = f"at projection {data.get('projection', 1.0):g}"
    if drawn and data["seed"] is None:
        raise ValueError(f"{path_of(data)}: goo wired {how} without a seed cannot be rebuilt")
    goo = Goo(
        count=data["count"],
        across=across_of(data),
        weight=None if data["random_weights"] else data["weight"],
        threshold=data["threshold"],
        seed=data["seed"],
        weight_range=tuple(data.get("weight_range", (-1.0, 1.0))),
        minimum_potential=data.get("minimum_potential", -1.0),
        scale_with_fan_in=data.get("scale_with_fan_in", True),
        projection=data.get("projection", 1.0),
        outputs=data.get("outputs"),
        wiring=wiring,
        scaling_factor=scaling_factor,
    )
    load_weights(goo, data)
    goo.epoch = data["epoch"]
    return goo



def path_of(data: dict) -> str:
    return data.get("path", "checkpoint")


def exploration_of(data: dict) -> str:
    """What a checkpoint was saved exploring by (§7.1): its setting, or the neuron rule where it carries none (§8.14).

    Refused (§12.2): an exploration the file does not name, and one its
    format contradicts -- format 3 is written under exploration at the
    synapse and format 2 under the neuron rule, so a file that says otherwise
    was written by no build of this code.
    """
    mode = data.get("exploration", "neuron")
    if mode not in EXPLORATIONS:
        raise ValueError(f"{path_of(data)}: unknown exploration {mode!r}; the file names {', '.join(EXPLORATIONS)} (§7.1)")
    if (mode == "synapse") != (data.get("format") == SYNAPSE_FORMAT):
        raise ValueError(f"{path_of(data)}: format {data.get('format')!r} with exploration {mode!r}: format "
                         f"{SYNAPSE_FORMAT} is written under exploration at the synapse and format {FORMAT} under the neuron "
                         "rule (§8.14), so this file is not one a checkpoint wrote")
    return mode


def load_weights(network: Goo, data: dict) -> None:
    """Copy a checkpoint's weights into `network`, which must have the same wiring.

    The network resumes under the exploration the checkpoint was saved under
    (§12.9): a fresh network takes it, with its settings, and a network
    already exploring at the synapse given a checkpoint of the neuron rule --
    or one that has run under the neuron rule given a checkpoint of the
    synapse's -- is refused, nothing mapping the one's bookkeeping onto the
    other's (Q17).
    """
    mode = exploration_of(data)
    if network.exploration == "synapse" and mode == "neuron":
        raise ValueError("this network explores at the synapse and the checkpoint was saved under the neuron rule: a run "
                         "is not resumed under the other exploration, nothing mapping E_j and the credit onto the gain "
                         "(§12.9)")
    wanted = {"across": across_of(data), "rows": data["rows"], "seed": data["seed"]}
    for key, value in wanted.items():
        if getattr(network, key) != value:
            raise ValueError(f"checkpoint {key} is {value!r} but the mesh has {getattr(network, key)!r}")
    if len(network.connections) != data["connections"] or len(data["weights"]) != data["connections"]:
        raise ValueError(
            f"checkpoint has {data['connections']} connections but the mesh has {len(network.connections)}"
        )
    # §12.11: neuron for neuron and synapse for synapse against a fresh build, and refused if they differ.
    # A file written before the digest existed carries none and keeps the count check alone.
    recorded = data.get("wiring_digest")
    if recorded is not None and recorded != wiring_digest(network):
        raise ValueError(
            "checkpoint's wiring differs from the mesh rebuilt from its seed: the same neuron count and the same "
            "number of synapses, but not the same synapses. §12.11 refuses this rather than loading weights that "
            "were scored against another topology"
        )
    for connection_id, weight in enumerate(data["weights"], start=1):
        network.connections[connection_id].weight = weight
    neurons = list(network.all_neurons())
    for neuron, threshold in zip(neurons, data.get("thresholds", [])):
        neuron.threshold = threshold
    for neuron, rate in zip(neurons, data.get("rates", [])):
        neuron.rate = rate
    _restore_clock(network, data)


def resume_teacher(teacher, data: dict) -> None:
    """Continue a Teacher from a checkpoint: its running statistics from the learning record, and its stream.

    The Teacher's stream is the exploration stream the decisions draw from
    (§7.3), so it goes on from the state the checkpoint carries rather than
    from the seed it was built with (§12.11). The reinforcement baseline is the
    learning record's, or where there is none -- a checkpoint the sweep driver
    wrote (docs/rust-sweep.py) -- the one it keeps beside its own fields (§9.3).
    Until September 25, 2026 the command line's resume reseeded the stream and
    dropped the driver's baseline, and was a new run rather than the same one.
    """
    resume_stream(teacher.rng, data)
    record = data.get("learning")
    if not record:
        if data.get("baseline") is not None:
            teacher.baseline = data["baseline"]
        return
    teacher.epochs = record["epochs"]
    teacher.total_reward = record["total_reward"]
    teacher.baseline = record["baseline"]
    teacher.average = record["average"]
    teacher.history = list(record.get("history", []))


def resume_stream(rng, data: dict) -> None:
    """Put an exploration stream -- a Teacher's, or an untrained run's -- where the checkpoint left it (§7.3, §12.11). A
    file written before the state was carried has none, and the stream stays as it was seeded."""
    _restore_stream(rng, data.get("explore_state"))


CLOCK = ("tau", "refractory", "hop", "bored_after", "rate_tau")  # the clock every neuron runs on, the Neuron class's
# attributes of those names; a checkpoint records each (§12.9), and a run resumed from it runs on them unless it names others


def clock_of(data: dict) -> dict:
    """The clock a checkpoint was saved running on (§12.9): TAU, the refractory period, the hop, bored-after and the rate
    read's window, by the names of CLOCK, each where the file records it. A file written before a field was carried has
    none of it, and a run resumed from it keeps its own; the hop of a file from before HOP is converted (hop_of)."""
    clock = {name: data[name] for name in ("tau", "refractory", "bored_after") if data.get(name) is not None}
    if data.get("hop") is not None or data.get("refractory_hops"):
        clock["hop"] = hop_of(data)
    if data.get("rate"):
        clock["rate_tau"] = data["rate"][1]
    return clock



def _restore_clock(network, data: dict) -> None:
    """Continue the clock: time, horizon, each neuron's potential and spikes, the synapse stamps, the signals in flight."""
    if exploration_of(data) == "synapse":
        # §12.9: the network resumes under the settings it was saved under, and kappa_i is recomputed from them, a rule
        # and not a state (§7.7, §12.10). Set before the state is loaded: set_exploration refuses a network that has
        # already run under the neuron rule (§12.9), and the state below is exploration at the synapse's own
        network.set_exploration("synapse", h0=data["synapse_hazard_rest"], family=data["synapse_hazard_family"],
                                scaling=data["synapse_hazard_scaling"], trace=data["trace_mode"])
        network.drive_steps = data["drive_steps"]  # 5.4b's DRIVE_STEPS, a run option (§12.9)
    # §12.11: the streams first, so the next draw is the one the uninterrupted run would have taken. A file written
    # before the states were carried has none, and a run resumed from it is a new run rather than the same one.
    _restore_stream(getattr(network, "_rng", None), data.get("network_state"))
    _restore_stream(getattr(network, "explore_rng", None), data.get("explore_state"))
    if data.get("input_at") is not None:
        network.input_at = data["input_at"]
    network.time = data.get("time", 0.0)
    network.interval = data.get("interval", network.interval)
    network.horizon = data.get("horizon", network.time)
    neurons = list(network.all_neurons())
    for neuron, floor in zip(neurons, data.get("floors", [])):
        neuron.minimum_potential = floor  # older checkpoints have none and keep the scalar they were built with
    for neuron, potential in zip(neurons, data.get("potentials", [])):
        neuron.potential = potential
    for neuron, fired_at in zip(neurons, data.get("fired_at", [])):
        neuron.fired_at = fired_at
    for neuron, last in zip(neurons, data.get("last_update", [])):
        neuron.last_update = last
    for neuron, previous in zip(neurons, data.get("previous_fired_at", [])):
        neuron.previous_fired_at = previous
    for neuron, spikes in zip(neurons, data.get("spikes", [])):
        neuron.spikes = spikes
    for connection_id, last in enumerate(data.get("last_signal", []), start=1):
        network.connections[connection_id].last_signal = last
    network.escape_delta = data.get("escape_delta", 0.0)  # escape noise (§5.2); older checkpoints ran the threshold
    for neuron, delta in zip(neurons, data.get("deltas", [])):
        neuron.delta = delta
        # the trace of §6.7 is kept under escape noise, the centred rule, or exploration at the synapse (§8.16)
        neuron.traced = delta > 0.0 or neuron.centred or neuron.synaptic
    from .network import escape_scale
    network.escape_scale = escape_scale(len(neurons))  # §5.2: the count's scaling of every hazard is a rule, recomputed not stored
    for neuron in neurons:
        neuron.escape_scale = network.escape_scale
    for neuron, since in zip(neurons, data.get("exposed_since", [])):
        neuron.exposed_since = since
    for connection_id, (trace, at) in enumerate(data.get("traces", []), start=1):
        network.connections[connection_id].trace, network.connections[connection_id].trace_at = trace, at
    for connection_id, noted in enumerate(data.get("notes", []), start=1):
        network.connections[connection_id].noted = noted
    for neuron, expectation in zip(neurons, data.get("expectations", [])):
        neuron.expectation = expectation
    for neuron, decisions in zip(neurons, data.get("decisions", [])):
        neuron.decisions = decisions
    for neuron, expected in zip(neurons, data.get("expected_since_spike", [])):
        neuron.expected = expected
    for neuron, gain in zip(neurons, data.get("gains", [])):
        neuron.gain = gain  # G_j (§8.16), under exploration at the synapse
    for neuron, count in zip(neurons, data.get("read_counts", [])):
        neuron.read_count = count  # this epoch's read-synapse escapes (§7.9), zeroed by the next epoch's reset
    network.schedule.clear()
    for entry in data.get("pending", []):  # (time, id), or (time, id, ventured) under exploration at the synapse (§7.9)
        network.schedule.signal(network.connections[entry[1]], entry[0], ventured=len(entry) > 2 and bool(entry[2]))
    for entry in data.get("waiting", []):  # the stimuli and charges still in the queue, in its order (§3.4)
        time, kind, i = entry[0], entry[1], entry[2]
        if kind == "stimulus":
            network.schedule.stimulus(neurons[i], time)
        elif kind == "charge":
            network.schedule.charge(neurons[i], entry[3], time)
        elif kind == "external":
            network.schedule.external(neurons[i], entry[3], time)
        else:
            raise ValueError(f"{path_of(data)}: unknown waiting event {kind!r}; a checkpoint writes stimulus, charge and "
                             "external")
    network.rule = data.get("rule", "local")  # a file written under a rule the specification dropped reads as local (§9.1)
    network.population = data.get("population", network.population)
    network.temperature = data.get("temperature", network.temperature)
    network.clock = data.get("clock", 0)
    network.pickiness = data.get("pickiness", network.pickiness)
    if data.get("quash"):
        network.quash_rate, network.quash_k = data["quash"]
    network.drive = data.get("drive", network.drive)
    if data.get("rate"):
        network.rate_on, Neuron.rate_tau = data["rate"]
    if data.get("input_rate"):
        network.input_rate, network.input_rate_off = data["input_rate"]

