#!/bin/bash
echo "=== PATCHING ONNXRuntime for execstack ==="
# already installed via packages.txt, but just in case
apt-get update -qq && apt-get install -y execstack -qq || echo "execstack already installed"

SO_PATH="$VIRTUAL_ENV/lib/python3.11/site-packages/onnxruntime/capi/onnxruntime_pybind11_state*.so"
if ls $SO_PATH 1> /dev/null 2>&1; then
    execstack -c $SO_PATH
    echo "execstack flag cleared on onnxruntime .so"
else
    echo "WARNING: onnxruntime .so not found — skip patch"
fi
echo "=== PATCH DONE ==="