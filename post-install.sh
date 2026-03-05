#!/bin/bash
set -e  # exit on any error

echo "=== POST-INSTALL SCRIPT STARTED ==="
echo "Current user: $(whoami)"
echo "Virtual env path: $VIRTUAL_ENV"
echo "Python version: $(python --version)"

echo "Step 1: apt update..."
apt-get update -qq || { echo "apt update FAILED"; exit 1; }
echo "apt update done"

echo "Step 2: install execstack..."
apt-get install -y execstack || { echo "install execstack FAILED"; exit 1; }
echo "execstack installed"

echo "Step 3: find onnxruntime .so..."
SO_FILE=$(ls $VIRTUAL_ENV/lib/python3.11/site-packages/onnxruntime/capi/onnxruntime_pybind11_state*.so 2>/dev/null || echo "NOT FOUND")
echo "SO_FILE = $SO_FILE"

if [ "$SO_FILE" = "NOT FOUND" ]; then
    echo "onnxruntime .so not found - skipping patch"
else
    echo "Step 4: clearing execstack flag..."
    execstack -c "$SO_FILE" || { echo "execstack -c FAILED"; exit 1; }
    echo "execstack cleared successfully"
fi

echo "=== POST-INSTALL SCRIPT FINISHED ==="