#!/bin/bash
echo "=== Patching onnxruntime for execstack ==="
apt-get update -qq && apt-get install -y execstack -qq
execstack -c $VIRTUAL_ENV/lib/python3.11/site-packages/onnxruntime/capi/onnxruntime_pybind11_state*.so
echo "Patch done - execstack cleared"