#!/bin/bash
export TMPDIR=/media/scott/data/tmp
cd /media/scott/data/lfm3-npu-tool
export LD_LIBRARY_PATH="/media/scott/data/ryzen-ai-venv/lib/python3.12/site-packages/flexml/flexml_extras/lib:/media/scott/data/ryzen-ai-venv/lib/python3.12/site-packages/voe/lib:/opt/xilinx/xrt/lib:$LD_LIBRARY_PATH"
find lib/lfm3_npu -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
pkill -9 -f cpu_batched_server 2>/dev/null || true
sleep 2
nohup bash -c 'python3 -B lib/lfm3_npu/cpu_batched_server.py --host 0.0.0.0 --port 1238' > var/log/cpu_batched.log 2>&1 &
echo "CPU server started with PID $!"
sleep 15
curl -s --max-time 5 http://0.0.0.0:1238/health 2>&1