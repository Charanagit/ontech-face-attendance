#!/bin/bash
set -e

echo "=== SETUP.SH STARTED ==="
echo "Python: $(python --version)"
echo "Pip: $(pip --version)"

echo "Force uninstall numpy..."
pip uninstall -y numpy || true

echo "Force install NumPy 1.26.4 wheel..."
pip install numpy==1.26.4 --no-cache-dir --force-reinstall --verbose

echo "Install rest of requirements..."
pip install -r requirements.txt --no-cache-dir --verbose

echo "=== SETUP.SH FINISHED ==="