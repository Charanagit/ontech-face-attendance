#!/bin/bash
echo "Patching onnxruntime .so to remove execstack flag..."
apt-get update -qq && apt-get install -y execstack -qq
execstack -c /home/adminuser/venv/lib/python3.11/site-packages/onnxruntime/capi/onnxruntime_pybind11_state*.so
echo "Patch complete - execstack cleared"