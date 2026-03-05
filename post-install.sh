#!/bin/bash
echo "================================================================="
echo "POST-INSTALL.SH IS RUNNING - START"
echo "Current working dir: $(pwd)"
echo "Home dir: $HOME"
echo "VIRTUAL_ENV: $VIRTUAL_ENV"
echo "PATH: $PATH"
echo "================================================================="

SO_GLOB="$VIRTUAL_ENV/lib/python3.11/site-packages/onnxruntime/capi/onnxruntime_pybind11_state*.so"

echo "Checking for onnxruntime shared object..."
if ls $SO_GLOB >/dev/null 2>&1; then
    echo "Found file(s):"
    ls -l $SO_GLOB
    echo "Running patchelf --clear-execstack..."
    patchelf --clear-execstack $SO_GLOB || echo "patchelf failed - check if installed"
    echo "Patch attempted."
else
    echo "ERROR: No onnxruntime .so found at $SO_GLOB"
    echo "List site-packages/onnxruntime/capi:"
    ls -l $VIRTUAL_ENV/lib/python3.11/site-packages/onnxruntime/capi || echo "dir not found"
fi

echo "================================================================="
echo "POST-INSTALL.SH FINISHED"
echo "================================================================="