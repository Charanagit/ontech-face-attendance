#!/bin/bash
echo "Forcing NumPy 1.26.4 before other installs..."
pip uninstall -y numpy
pip install numpy==1.26.4 --no-cache-dir --force-reinstall
pip install -r requirements.txt --no-cache-dir
echo "Setup complete - NumPy locked to 1.26.4"