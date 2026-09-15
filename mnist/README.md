# mnist

The MNIST handwritten digits (LeCun, Cortes and Burges), the data folder of the
mnist problem (AUTHORITY.md §8). Two IDX files, gzipped, as distributed: the
60,000 training images with their labels, each image 28 × 28 unsigned bytes.
**The test split is not here and is not to be fetched** (Byron, September 15,
2026: "we do not dare touch the test split"); the network keeps living (§7),
and there is no held-out set to measure it on.

Fetched September 15, 2026 from the mirror TensorFlow uses,
`https://storage.googleapis.com/cvdf-datasets/mnist/<file>`. The files are not
in git (see `.gitignore`); `walnutbutter.mnist.fetch()` downloads them again
and checks these sizes and SHA-256 sums:

| file | bytes | sha256 |
|---|---|---|
| train-images-idx3-ubyte.gz | 9,912,422 | 440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609 |
| train-labels-idx1-ubyte.gz | 28,881 | 3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c |

`walnutbutter.mnist` reads them with nothing but gzip and numpy.
