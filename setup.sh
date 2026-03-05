#!/bin/bash
set -e  # exit on error

echo "=== Custom setup script started ==="
echo "Python version: $(python --version)"
echo "Pip version: $(pip --version)"

echo "Uninstalling any existing numpy..."
pip uninstall -y numpy || true

echo "Installing pinned numpy==1.26.4..."
pip install numpy==1.26.4 --no-cache-dir --force-reinstall --verbose

echo "Installing remaining requirements..."
pip install -r requirements.txt --no-cache-dir --verbose

echo "=== Setup complete - NumPy locked to 1.26.4 ==="