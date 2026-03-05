#!/bin/bash
echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
echo "POST-INSTALL.SH IS EXECUTING RIGHT NOW - PATCH ATTEMPT"
echo "Working directory: $(pwd)"
echo "VIRTUAL_ENV: $VIRTUAL_ENV"
echo "Python: $(python --version 2>&1)"
echo "Looking for onnxruntime .so..."
SO_GLOB="$VIRTUAL_ENV/lib/python3.11/site-packages/onnxruntime/capi/onnxruntime_pybind11_state*.so"
if ls $SO_GLOB >/dev/null 2>&1; then
    echo "FOUND: $(ls -l $SO_GLOB)"
    patchelf --clear-execstack $SO_GLOB && echo "PATCH SUCCESS - execstack cleared" || echo "patchelf FAILED"
else
    echo "NOT FOUND - onnxruntime .so missing"
    echo "Listing capi dir:"
    ls -l $VIRTUAL_ENV/lib/python3.11/site-packages/onnxruntime/capi 2>&1 || echo "dir not found"
fi
echo "POST-INSTALL.SH ENDED"
echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"