# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import inspect
import os
import pytest
import subprocess
import unittest
import torch
from mnist_poptorch import accuracy


def run_poptorch_mnist(**kwargs):
    cwd = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
    cmd = ["python3", 'mnist_poptorch.py']
    try:
        out = subprocess.check_output(
            cmd, cwd=cwd, stderr=subprocess.PIPE).decode("utf-8")
    except subprocess.CalledProcessError as e:
        print(f"TEST FAILED")
        print(f"stdout={e.stdout.decode('utf-8',errors='ignore')}")
        print(f"stderr={e.stderr.decode('utf-8',errors='ignore')}")
        raise
    return out


class TestPoptorchMNIST(unittest.TestCase):
    def test_accuracy_calculation(self):
        pred = torch.tensor([[0.9, 0.05, 0.05],
                             [0.1, 0.5, 0.4],
                             [0.6, 0.01, 0.49],
                             [0.09, 0.11, 0.8]])
        label = torch.tensor([0, 1, 2, 2])
        acc = accuracy(pred, label)
        self.assertEqual(acc, 75)

    @pytest.mark.ipus(1)
    def test_test_final_training_accuracy(self):
        out = run_poptorch_mnist()
        final_acc = 0.0
        for line in out.split('\n'):
            if line.find('Accuracy on test set:') != -1:
                final_acc = float(line.split(": ")[-1].strip()[:-1])
                break
        self.assertGreater(final_acc, 89)
        self.assertLess(final_acc, 99.9)
