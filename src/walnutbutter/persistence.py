"""Saving and restoring what the network has learned.

A checkpoint is a JSON file holding every connection's weight (by ID) together
with what is needed to rebuild the identical network: count, the wiring
rule, threshold and seed. Loading rebuilds it from
those settings
and copies the weights back in, so a run that took hours can be continued or
inspected later.
"""

from __future__ import annotations

import json
from pathlib import Path

from .constants import GOO_SCALING_FACTOR
from .goo import Goo, scaled_projection
from .network import Network
from .neuron import Neuron

FORMAT = 2  # 1 held a permutation, an input coding, an error-correcting code and a flip; §5.2 dropped all four


def checkpoint(network: Network, path: str | Path, teacher=None) -> dict:
    """Write the network's weights and settings to `path`. Returns what was written."""
    engine = getattr(network, "engine", "objects")
    if engine == "arrays":
        network.sync_to_mesh()  # the checkpoint is written from the mesh, whichever engine ran it
        network = network.mesh
    data = {
        "format": FORMAT,
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
        "tau": Neuron.tau,
        "isi_factor": Neuron.isi_factor,  # §0.2: whether the single-spike rule's charges were weighed; a checkpoint
        "target_isi": Neuron.target_isi,  # without the key is from before the factor, and ran without it
        "refractory": Neuron.refractory,
        "refractory_hops": Neuron.refractory_hops,
        "bored_after": Neuron.bored_after,
        "horizon": network.horizon,  # the time the schedule has run to
        "connections": len(network.connections),
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
        "pending": [[time, connection.id] for time, connection in network.schedule.pending()],
        "rule": network.rule,  # which rule pays at the read: "reinforce", or "local" for none (§9.1)
    }
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


def across_of(data: dict) -> int:
    """The count across a checkpoint's input zone: "across", or "columns" in files written before the rename."""
    return data["across"] if "across" in data else data["columns"]


def read_checkpoint(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text())
    if data.get("format") != FORMAT:
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


def load_weights(network: Goo, data: dict) -> None:
    """Copy a checkpoint's weights into `network`, which must have the same wiring."""
    wanted = {"across": across_of(data), "rows": data["rows"], "seed": data["seed"]}
    for key, value in wanted.items():
        if getattr(network, key) != value:
            raise ValueError(f"checkpoint {key} is {value!r} but the mesh has {getattr(network, key)!r}")
    if len(network.connections) != data["connections"] or len(data["weights"]) != data["connections"]:
        raise ValueError(
            f"checkpoint has {data['connections']} connections but the mesh has {len(network.connections)}"
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
    """Continue a Teacher's running statistics from a checkpoint's learning record, if any."""
    record = data.get("learning")
    if not record:
        return
    teacher.epochs = record["epochs"]
    teacher.total_reward = record["total_reward"]
    teacher.baseline = record["baseline"]
    teacher.average = record["average"]
    teacher.history = list(record.get("history", []))



def _restore_clock(network, data: dict) -> None:
    """Continue the clock: time, horizon, each neuron's potential and spikes, the synapse stamps, the signals in flight."""
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
        neuron.traced = delta > 0.0 or neuron.centred  # the trace of §6.7 is kept under escape noise
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
    network.schedule.clear()
    for time, connection_id in data.get("pending", []):
        network.schedule.signal(network.connections[connection_id], time)
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

