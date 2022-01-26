# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import os
from pathlib import Path
import pytest

# NOTE: The import below is dependent on 'pytest.ini' in the root of
# the repository
from tutorials_tests.testing_util import SubProcessChecker
from tutorials_tests.xdist_util import lock

working_path = Path(__file__).parent


class TestBuildAndRun(SubProcessChecker):

    @lock(os.path.join(working_path, "binary.lock"))
    def setUp(self):
        ''' Compile the complete version of the tutorial code '''
        self.run_command("make all", working_path, [])

    @pytest.mark.category1
    def test_run_ipu_model(self):
        ''' Check that the complete version of the tutorial code
            for the IPU Model runs '''

        self.run_command("./tut1_ipu_model_complete",
                         working_path,
                         ["Program complete", "h3 data:", "0 1 1.5 2",
                          "v4-1: {10,11,12,13,14,15,16,17,18,19}"])

    @pytest.mark.category1
    @pytest.mark.ipus(1)
    def test_run_ipu_hardware(self):
        ''' Check that the complete version of the tutorial code
            for the IPU hardware runs '''
        self.run_command("./tut1_ipu_hardware_complete",
                         working_path,
                         ["Attached to IPU", "Program complete",
                          "h3 data:", "0 1 1.5 2",
                          "v4-1: {10,11,12,13,14,15,16,17,18,19}"])
