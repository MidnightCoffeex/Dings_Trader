#!/bin/bash
# Auto-restart wrapper for dings-trader API

LOG_FILE="/tmp/dings_api.log"
PID_FILE="/tmp/dings_api.pid"

cd /home/maxim/.openclaw/workspace/projects/dings-trader/ml
source ../TraderHimSelf/venv/bin/activate

# Kill existing
if [ -f "$PID_FILE" ]; then
    old_pid=$(cat "$PID_FILE")
    kill $old_pid 2>/dev/null
    sleep 1
fi

echo "[$(date)] Starting API server..." >> "$LOG_FILE"

# Start with auto-restart
while true; do
    echo "[$(date)] Starting uvicorn..." >> "$LOG_FILE"
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload >> "$LOG_FILE" 2>&1 &
    PID=$!
    echo $PID > "$PID_FILE"
    wait $PID
    EXIT_CODE=$?
    echo "[$(date)] API crashed with exit code $EXIT_CODE, restarting in 5s..." >> "$LOG_FILE"
    sleep 5
done
