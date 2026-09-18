"""Input patterns for the network.

The network's input is its input zone (AUTHORITY.md §5.1). A pattern is one
boolean per place in the zone; the neurons whose bit is 1 are driven.

An input is complement-coded and coded no other way (§5.2): k raw bits
followed by their negations, 2k coded bits onto 2k input neurons, so exactly
half the zone is driven whatever the raw bits are. Place i of the zone shows
place i of the coded pattern -- there is no permutation -- and the pattern
presented is the pattern the read is scored against: nothing corrupts an
input on the way in.
"""

from __future__ import annotations

import random
from typing import Sequence


def random_bits(count: int, seed: int | None = None) -> list[bool]:
    """`count` independent fair coin flips, reproducible with `seed`."""
    rng = random.Random(seed)
    return [rng.random() < 0.5 for _ in range(count)]


def complement_code(bits: Sequence[bool]) -> list[bool]:
    """The bits followed by their complements: [b0, b1, ...] -> [b0, b1, ..., not b0, not b1, ...]."""
    bits = [bool(b) for b in bits]
    return bits + [not b for b in bits]


def parse_bits(text: str) -> list[bool]:
    """Turn a string such as "101100" into bits. Spaces are ignored."""
    cleaned = text.replace(" ", "")
    if not cleaned or any(ch not in "01" for ch in cleaned):
        raise ValueError(f"input must be a string of 0s and 1s, got {text!r}")
    return [ch == "1" for ch in cleaned]


def format_bits(bits: Sequence[bool]) -> str:
    return "".join("1" if b else "0" for b in bits)


# --- error-correcting codes -----------------------------------------------------------

from dataclasses import dataclass
