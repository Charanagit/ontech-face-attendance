#!/bin/bash
echo "Forcing NumPy 1.26.4 wheel from direct URL..."
pip uninstall -y numpy
pip install https://files.pythonhosted.org/packages/b3/2e/2c6f94c6f2c6a2715c4706f9b36a5ace817023d949f4d6a360f09d9b5f9/numpy-1.26.4-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl --no-cache-dir --force-reinstall
pip install -r requirements.txt --no-cache-dir
echo "Setup complete - NumPy locked to 1.26.4 wheel"