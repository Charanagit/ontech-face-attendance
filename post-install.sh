#!/bin/bash
echo "=== PATCHING ONNXRuntime executable stack with patchelf ==="

SO_FILE="$VIRTUAL_ENV/lib/python3.11/site-packages/onnxruntime/capi/onnxruntime_pybind11_state*.so"

if ls $SO_FILE >/dev/null 2>&1; then
    patchelf --clear-execstack $SO_FILE
    echo "execstack flag CLEARED using patchelf on onnxruntime .so"
else
    echo "WARNING: onnxruntime .so not found - patch skipped"
fi

echo "=== PATCH DONE ==="