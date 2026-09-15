# mnist

The MNIST handwritten digits (LeCun, Cortes and Burges), the data folder of the
mnist problem (AUTHORITY.md §8). Four IDX files, gzipped, as distributed:
60,000 training images with labels and 10,000 test images with labels, each
image 28 × 28 unsigned bytes.

Fetched September 15, 2026 from the mirror TensorFlow uses,
`https://storage.googleapis.com/cvdf-datasets/mnist/<file>`. The files are not
in git (see `.gitignore`); `walnutbutter.mnist.fetch()` downloads them again
and checks these sizes and SHA-256 sums:

| file | bytes | sha256 |
|---|---|---|
| train-images-idx3-ubyte.gz | 9,912,422 | 440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609 |
| train-labels-idx1-ubyte.gz | 28,881 | 3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c |
| t10k-images-idx3-ubyte.gz | 1,648,877 | 8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6 |
| t10k-labels-idx1-ubyte.gz | 4,542 | f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6 |

`walnutbutter.mnist` reads them with nothing but gzip and numpy.
