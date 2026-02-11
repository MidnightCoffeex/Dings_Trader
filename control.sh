#!/bin/bash
# Dings Trader Control Center (PM2 based)
# Usage: ./control.sh [start|stop|restart|status|logs|build]

ECO_FILE="ecosystem.config.js"
UI_DIR="ui"

case "$1" in
  start)
    echo "🚀 Starting Dings Trader via PM2..."
    pm2 start $ECO_FILE
    pm2 save
    ;;
  
  stop)
    echo "🛑 Stopping Dings Trader..."
    pm2 stop $ECO_FILE
    ;;
    
  delete)
    echo "🗑️ Deleting processes..."
    pm2 delete $ECO_FILE
    ;;

  restart)
    echo "🔄 Full Restart sequence..."
    # 1. Stop UI first to release port (safe side)
    pm2 stop dt-ui
    
    # 2. Rebuild UI
    echo "🏗️ Rebuilding Frontend..."
    cd $UI_DIR
    rm -rf .next
    npm run build
    cd ..
    
    # 3. Restart everything
    echo "🚀 Reloading PM2..."
    pm2 restart $ECO_FILE
    ;;

  build)
    echo "🏗️ Rebuilding Frontend ONLY..."
    cd $UI_DIR
    rm -rf .next
    npm run build
    ;;

  status)
    pm2 status
    ;;

  logs)
    pm2 logs
    ;;

  *)
    echo "Usage: $0 {start|stop|restart|build|status|logs|delete}"
    exit 1
    ;;
esac
