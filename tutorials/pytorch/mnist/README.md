# Spell on Graphcore IPU

---
## PyTorch(PopTorch) MNIST Training Demo <a href="https://web.spell.ml/workspace_create?workspaceName=deeplab-voc-2012&githubUrl=https%3A%2F%2Fgithub.com%2Fspellml%2Fdeeplab-voc-2012"><img src=https://spell.ml/badge.svg height=20px/></a>

This example demonstrates how to train a network on the MNIST dataset using PopTorch on the Spell platform. 

### File structure

* `mnist_poptorch.py` The main file.
* `test_mnist.py` Test file.
* `requirements.txt` Pip dependencies needed for tutorial. 

### How to use this demo

1) Execute `spell run` to train the dataset

    The Poplar SDK and PopTorch framework are by default installed on a Spell default IPU image. You can begin training your model in a single CLI command! 


       spell run --machine-type IPU16 \ 
            --pip-req requirements.txt \
            "python3 mnist_poptorch.py"


2) View resources and outputs from runs

       [Placehodler image]
       

3) Run the test script to see your results

       spell run --machine-type IPU16 \ 
            --pip-req requirements.txt \
            "python3 test_mnist.py"

#### Options
The program has a few command-line options:

`-h` Show usage information.

`--batch-size`        Sets the batch size for training.

`--batches-per-step`  Number on mini-batches to perform on the device before returning to the host.

`--test-batch-size`   Sets the batch size for inference.

`--epochs`            Number of epoch to train for.

`--lr`                Learning rate of the optimizer.

