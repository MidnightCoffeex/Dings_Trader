#!/bin/bash
# Dings-Trader Dashboard Production Server
# Auto-restart wrapper script

PROJECT_DIR="/home/maxim/.openclaw/workspace/projects/dings-trader/ui"
PORT=3000
HOST="0.0.0.0"

cd "$PROJECT_DIR" || exit 1

while true; do
    echo "[$(date)] Starting dings-trader dashboard..."
    
    # Kill any existing process on port 3000
    fuser -k ${PORT}/tcp 2>/dev/null
    sleep 1
    
    # Start the server
    npm start -- --hostname $HOST --port $PORT
    
    EXIT_CODE=$?
    echo "[$(date)] Server exited with code $EXIT_CODE"
    
    # Wait before restart
    echo "[$(date)] Restarting in 3 seconds..."
    sleep 3
done
