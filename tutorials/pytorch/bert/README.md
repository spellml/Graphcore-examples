Tutorial on BERT Fine-tuning on IPU
===================================

This tutorial demonstrates how to fine-tune a pre-trained BERT model with PyTorch on the Graphcore IPU-POD16 system. It uses a BERT-Large model and fine-tunes it on the SQuADv1 Question/Answering task. The tutorial is in [Fine-tuning-BERT.ipynb](./Fine-tuning-BERT.ipynb).

### File structure

* `README.md` This file
* `Fine-tuning-BERT.ipynb` The tutorial jupyter notebook
* `Fine-tuning-BERT.py` Python script of the jupyter notebook
* `requirements.txt` Required packages
* `tests/test_finetuning_notebook.py` Script for testing this tutorial
* `tests/requirements.txt` Required packages for the tests

### How to use the notebook tutorial

1.	Create a Spell Workspace
2.	Initialise BERT Code
3.	Setup Poplar SDK Environment
4.	Run the BERT Finetuning notebook on IPU

Follow the [Spell Quickstart](https://spell.ml/docs/quickstart/) to install the Spell cli and login to your account.

```
pip install spell
spell login
```

Run the following command on your terminal to create and launch the workspace in the web browser.
```
spell jupyter bert --machine-type IPUx16 \
--github-url "https://github.com/spellml/Graphcore-examples" \
--docker-image "graphcore/pytorch:latest" --lab
```

On the workspace view, change your directory to `tutorials/pytorch/bert` and open the Fine-tuning-BERT.ipynb notebook.
On the first cell, make sure to  install python packages from requirements files by running:

`%pip install -r requirements.txt`

Once these steps are done, you can now click-through the notebook to follow the process of fine-tuning a pre-trained BERT model using Graphcore IPU-POD16 system in Spell.

### License

The file `squad_preprocessing.py` is based on code from Hugging Face licensed under Apache 2.0 so is distributed under the same license (see the LICENSE file in this directory for more information).

The rest of the code in this example is licensed under the MIT license - see the LICENSE file at the top level of this repository.

