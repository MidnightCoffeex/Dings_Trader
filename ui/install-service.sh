#!/bin/bash
# Install dings-trader as systemd service

echo "Installing dings-trader systemd service..."

# Copy service file
sudo cp /home/maxim/.openclaw/workspace/projects/dings-trader/ui/dings-trader.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable and start service
sudo systemctl enable dings-trader
sudo systemctl start dings-trader

echo "Service installed! Commands:"
echo "  sudo systemctl status dings-trader"
echo "  sudo systemctl start dings-trader"
echo "  sudo systemctl stop dings-trader"
echo "  sudo systemctl restart dings-trader"
