#!/bin/bash
# Start paper trading inference loop (PPO v1)

cd /home/maxim/.openclaw/workspace/projects/dings-trader/ml
source ../TraderHimSelf/venv/bin/activate

# Kill existing loops
pkill -f "paper_inference_loop.py" 2>/dev/null
sleep 2

echo "Starting Paper Trading Loop (paper_ppo_v1)..."

python paper_inference_loop.py \
    --model-id paper_ppo_v1 \
    --symbol BTCUSDT \
    --interval 60 \
    --create-account > /tmp/paper_ppo_v1.log 2>&1 &

echo "Started paper_ppo_v1 - PID: $!"
echo "Log: /tmp/paper_ppo_v1.log"
