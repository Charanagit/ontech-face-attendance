#!/bin/bash
set -e

echo "=== POST-INSTALL.SH STARTED ==="
echo "Current dir: $(pwd)"
echo "VIRTUAL_ENV: $VIRTUAL_ENV"
echo "Python: $(python --version)"
echo "Pip: $(pip --version)"

SO_FILE="$VIRTUAL_ENV/lib/python3.11/site-packages/onnxruntime/capi/onnxruntime_pybind11_state*.so"

echo "Looking for onnxruntime .so at: $SO_FILE"

if ls $SO_FILE >/dev/null 2>&1; then
    echo "Found .so file(s): $(ls $SO_FILE)"
    patchelf --clear-execstack $SO_FILE
    echo "patchelf --clear-execstack ran successfully"
else
    echo "ERROR: onnxruntime .so NOT FOUND - check if onnxruntime installed correctly"
fi

echo "=== POST-INSTALL.SH FINISHED ==="