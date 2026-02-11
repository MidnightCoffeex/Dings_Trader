#!/bin/bash
cd /home/maxim/.openclaw/workspace/projects/dings-trader/ml
./.venv/bin/python live_inference.py --model-id v2 --model-file model_v2.1.joblib --loop --sleep 300
