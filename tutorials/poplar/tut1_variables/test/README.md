This directory contains tests for the Poplar tutorial 1

The tests use `pytest` and helper utilities in the `utils` directory
in the root of this repository.

To run the tests, create a python3 virtual environment and install the
requirements:

    pip install -r requirements.txt

Source the `enable.sh` script for Poplar as described in the
Getting Started Guide for your IPU system.

Then run the tests using pytest:

    pytest
