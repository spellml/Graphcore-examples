# Spell on Graphcore IPU

---
## PyTorch(PopTorch) MNIST Training Demo <a href="https://web.spell.ml/workspace_create?workspaceName=deeplab-voc-2012&githubUrl=https%3A%2F%2Fgithub.com%2Fspellml%2Fdeeplab-voc-2012"><img src=https://spell.ml/badge.svg height=20px/></a>

This example demonstrates how to train a network on the MNIST dataset using PopTorch on the Spell platform.

### File structure

* `mnist_pytorch.py` The main file.
* `requirements.txt` Pip dependencies needed for tutorial.

### How to use this demo

1) Execute `spell run` to train the dataset

The Poplar SDK and PopTorch framework are by default installed on a Spell default IPU image. You can begin training your model in a single CLI command!

`--machine-type` Specifies IPU usage (but can be replaced with other compute such as V100s)

`--pip-req` Specifies a `requirements.txt` file to configure requirements during environment setup

`--docker-image` Sets Graphcore docker image for use; image should be switched depending on framework. See more information [here](https://www.docker.com/blog/graphcore-poplar-sdk-container-images-now-available-on-docker-hub/)

	spell run --machine-type IPUx16 --pip-req requirements.txt --docker-image graphcore/pytorch:latest 'python3 mnist_pytorch.py'

#### Options
The program has a few command-line options:

`-h` Show usage information.

`--batch-size`        Sets the batch size for training.

`--batches-per-step`  Number on mini-batches to perform on the device before returning to the host.

`--test-batch-size`   Sets the batch size for inference.

`--epochs`            Number of epoch to train for.

`--lr`                Learning rate of the optimizer.
