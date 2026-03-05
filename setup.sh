#!/bin/bash
set -e

echo "=== SETUP.SH STARTED - FORCING NUMPY WHEEL ==="
echo "Python version: $(python --version)"
echo "Pip version: $(pip --version)"

echo "Uninstall any existing numpy..."
pip uninstall -y numpy || true

echo "Installing numpy 1.26.4 wheel directly from official URL..."
pip install https://files.pythonhosted.org/packages/8d/3b/8d2b3e4b5a8d8e8d8e8d8e8d8e8d8e8d8e8d8e8d8e8d8e8d8e8d8e8d8e8d/numpy-1.26.4-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl --no-cache-dir --force-reinstall --verbose

echo "Installing remaining requirements..."
pip install -r requirements.txt --no-cache-dir --verbose

echo "=== SETUP.SH FINISHED - NumPy should be wheel-installed ==="