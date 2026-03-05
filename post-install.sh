#!/bin/bash
echo "=== PATCHING ONNXRuntime executable stack ==="
apt-get update -qq && apt-get install -y execstack -qq || true

SO_FILE="$VIRTUAL_ENV/lib/python3.11/site-packages/onnxruntime/capi/onnxruntime_pybind11_state*.so"

if ls $SO_FILE >/dev/null 2>&1; then
    execstack -c $SO_FILE
    echo "execstack flag CLEARED on onnxruntime .so"
else
    echo "WARNING: onnxruntime .so not found - patch skipped"
fi

echo "=== PATCH COMPLETE ==="