"""MNIST as the substance sees it: 28 x 28 handwritten digits made 14 x 14 bits (AUTHORITY.md §8, the mnist problem).

Byron, September 15, 2026, setting the task up: its own data folder, `mnist/`,
holding the training split's two IDX files as distributed (LeCun, Cortes and
Burges; [6] in BIBLIOGRAPHY.md) -- and not the test split: "we do not dare
touch the test split" (Byron, the same day), so it is not fetched, not read,
and not a split this module knows. Each image is averaged over 2 x 2 blocks to 14 x 14 and each
block on iff its mean intensity is at least half of full; 196 raw bits on 196
input neurons, coding raw, no permutation, the rate drive of §4.3 as it is.
The labels ride with the patterns: the stream a run is given is a seeded
shuffle of a split, cycling, with the label of each image beside it, and the
class critic (learning.class_accuracy) reads the label off the network.

Nothing here needs more than gzip and numpy. `fetch()` downloads the files
from the mirror TensorFlow uses and checks their sizes and SHA-256 sums.
"""

from __future__ import annotations

import gzip
import hashlib
import random
import struct
import urllib.request
from pathlib import Path

import numpy as np

FOLDER = Path(__file__).resolve().parent.parent.parent / "mnist"  # the data folder, beside src/
MIRROR = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = {  # split -> (images, labels): each (file name, bytes, sha256), as fetched September 15, 2026
    "train": (
        ("train-images-idx3-ubyte.gz", 9_912_422, "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609"),
        ("train-labels-idx1-ubyte.gz", 28_881, "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c"),
    ),
    # the test split (t10k-*) is deliberately absent: "we do not dare touch the test split" (Byron, September 15, 2026)
}
SIDE = 28  # pixels per side as distributed
BLOCK = 2  # the block averaged to one bit: 14 x 14 of them
BITS = (SIDE // BLOCK) ** 2  # 196 raw bits an image
CLASSES = 10
THRESHOLD = 0.5  # a block is on iff its mean intensity is at least this fraction of full ("half", Byron)


def available(folder: Path | str = FOLDER) -> bool:
    """True when the training split's two files are in the folder."""
    return all((Path(folder) / name).exists() for pair in FILES.values() for name, _, _ in pair)


def fetch(folder: Path | str = FOLDER, verbose: bool = True) -> None:
    """Download any file that is missing, then check every file's size and SHA-256 against the record."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for pair in FILES.values():
        for name, size, digest in pair:
            path = folder / name
            if not path.exists():
                if verbose:
                    print(f"fetching {MIRROR}{name} ({size:,} bytes)")
                urllib.request.urlretrieve(MIRROR + name, path)
            check(path, size, digest)


def check(path: Path, size: int, digest: str) -> None:
    """Refuse a file whose size or SHA-256 is not the recorded one."""
    actual = path.stat().st_size
    if actual != size:
        raise ValueError(f"{path.name}: {actual:,} bytes, expected {size:,}")
    found = hashlib.sha256(path.read_bytes()).hexdigest()
    if found != digest:
        raise ValueError(f"{path.name}: sha256 {found}, expected {digest}")


def read_idx(path: Path | str) -> np.ndarray:
    """One IDX file, gzipped, as the array it holds: labels (N,) or images (N, 28, 28), unsigned bytes."""
    with gzip.open(path, "rb") as handle:
        raw = handle.read()
    zero, kind, dims = struct.unpack(">HBB", raw[:4])
    if zero != 0 or kind != 0x08:
        raise ValueError(f"{path}: not an IDX file of unsigned bytes (magic {raw[:4].hex()})")
    shape = struct.unpack(">" + "I" * dims, raw[4:4 + 4 * dims])
    data = np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * dims)
    if data.size != int(np.prod(shape)):
        raise ValueError(f"{path}: {data.size:,} bytes of data for a shape of {shape}")
    return data.reshape(shape)


def load(split: str = "train", folder: Path | str = FOLDER) -> tuple[np.ndarray, np.ndarray]:
    """The split's images (N, 28, 28) and labels (N,), as distributed."""
    if split not in FILES:
        raise ValueError(f"split must be one of {', '.join(FILES)}, got {split!r}")
    if not available(folder):
        raise FileNotFoundError(f"MNIST is not in {folder}: run walnutbutter.mnist.fetch() (see mnist/README.md)")
    (images, _, _), (labels, _, _) = FILES[split]
    x, y = read_idx(Path(folder) / images), read_idx(Path(folder) / labels)
    if len(x) != len(y):
        raise ValueError(f"{split}: {len(x):,} images and {len(y):,} labels")
    return x, y


def downsample(images: np.ndarray, block: int = BLOCK) -> np.ndarray:
    """Average each `block` x `block` patch: (N, 28, 28) bytes to (N, 14, 14) means in [0, 1]."""
    n, h, w = images.shape
    if h % block or w % block:
        raise ValueError(f"{h} x {w} does not divide into {block} x {block} blocks")
    patches = images.reshape(n, h // block, block, w // block, block).astype(np.float64) / 255.0
    return patches.mean(axis=(2, 4))


def binarize(means: np.ndarray, threshold: float = THRESHOLD) -> np.ndarray:
    """A block is on iff its mean is at least `threshold`: (N, 14, 14) means to (N, 196) bits, row-major."""
    return (means >= threshold).reshape(len(means), -1)


def bits_of(split: str = "train", folder: Path | str = FOLDER) -> tuple[np.ndarray, np.ndarray]:
    """The split as the network sees it: (N, 196) bits and (N,) labels."""
    images, labels = load(split, folder)
    return binarize(downsample(images)), labels.astype(np.int64)


class Patterns:
    """A split's bits in a fixed order, one pattern a time, without 60,000 Python lists in memory.

    Indexing gives the pattern as a list of bools, which is what
    `Network.set_input_bits` takes; `len` is the split's size, and a run
    longer than that cycles, as §4.5 says a stream does.
    """

    def __init__(self, bits: np.ndarray, order: np.ndarray):
        self._bits, self._order = bits, order

    def __len__(self) -> int:
        return len(self._order)

    def __getitem__(self, k: int) -> list[bool]:
        return self._bits[self._order[k]].tolist()


def stream(split: str = "train", seed: int = 1, folder: Path | str = FOLDER) -> tuple[Patterns, list[int]]:
    """The split in a seeded shuffle, with the labels beside it: the input stream of the mnist problem (§4.5, §8).

    The order comes from `random.Random(f"walnutbutter mnist {split} {seed}")`,
    a stream of its own as the raw-bit stream's is, so seed s shows the same
    images in the same order to any network. Every image of the split is in
    the order once; a run longer than the split goes round again.
    """
    bits, labels = bits_of(split, folder)
    order = list(range(len(labels)))
    random.Random(f"walnutbutter mnist {split} {seed}").shuffle(order)
    order = np.array(order, dtype=np.int64)
    return Patterns(bits, order), labels[order].tolist()


_pixel_statistics = None  # (P(coded input on), P(coded input on | class)) over the training split, computed once


def pixel_statistics(clock: int = 0, folder: Path | str = FOLDER) -> tuple[np.ndarray, np.ndarray]:
    """P(coded input on) and P(coded input on | class), (clock + 2 * BITS,) and (CLASSES, clock + 2 * BITS), over the split.

    The coded input is the network's input zone in order: `clock` neurons
    always on, the 196 on-off bits, their 196 complements (§4.3, §8).
    """
    global _pixel_statistics
    if _pixel_statistics is None or _pixel_statistics[0] != clock:
        bits, labels = bits_of("train", folder)
        coded = np.concatenate([np.ones((len(bits), clock), dtype=bool), bits, ~bits], axis=1)
        on = coded.mean(0)
        on_given = np.stack([coded[labels == k].mean(0) for k in range(CLASSES)])
        _pixel_statistics = (clock, on, on_given)
    return _pixel_statistics[1], _pixel_statistics[2]


def supervised_direction(grid, folder: Path | str = FOLDER):
    """The supervised direction of a network's input-to-output synapses (§8): a function of the engine's edges.

    For a synapse from input i onto an output of class k the direction is
    P(coded input i on | class k) - P(coded input i on), the sign a linear
    classifier's gradient has on average; every other synapse is outside the
    mask. `fast.train(direction=...)` takes it, and records the estimator's
    correlation with it over the run.
    """
    on, on_given = pixel_statistics(grid.clock, folder)
    inputs = {neuron: i for i, neuron in enumerate(grid.input_row())}
    outputs = {neuron: k // grid.population for k, neuron in enumerate(grid.output_row())}

    def direction(edges):
        d = np.zeros(len(edges))
        mask = np.zeros(len(edges), dtype=bool)
        for e, c in enumerate(edges):
            i, k = inputs.get(c.source), outputs.get(c.target)
            if i is not None and k is not None:
                d[e] = on_given[k, i] - on[i]
                mask[e] = True
        return d, mask

    return direction
