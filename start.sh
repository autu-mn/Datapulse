#!/bin/bash
# OpenVista Service Launcher (Linux/macOS)
# Quick start script after installation

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "  ================================================================="
echo "  |   OpenVista Service Launcher                                 |"
echo "  ================================================================="
echo ""

# Check if services are already running
BACKEND_RUNNING=false
FRONTEND_RUNNING=false

if curl -s http://localhost:5001 > /dev/null 2>&1; then
    BACKEND_RUNNING=true
    echo "  [OK] Backend is running"
else
    echo "  [ ] Backend not running"
fi

if curl -s http://localhost:5173 > /dev/null 2>&1; then
    FRONTEND_RUNNING=true
    echo "  [OK] Frontend is running"
else
    echo "  [ ] Frontend not running"
fi

echo ""

if [ "$BACKEND_RUNNING" = true ] && [ "$FRONTEND_RUNNING" = true ]; then
    echo "  All services are running!"
    echo ""
    echo "  Opening browser..."
    
    if [[ "$OSTYPE" == "darwin"* ]]; then
        open "http://localhost:5173"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        xdg-open "http://localhost:5173" 2>/dev/null || echo "  Please open http://localhost:5173 in your browser"
    fi
    exit 0
fi

# Start backend
if [ "$BACKEND_RUNNING" = false ]; then
    echo "  Starting backend..."
    cd "$SCRIPT_DIR/backend"
    nohup python3 app.py > /dev/null 2>&1 &
    # Or use: python app.py &
    cd "$SCRIPT_DIR"
    sleep 3
fi

# Start frontend
if [ "$FRONTEND_RUNNING" = false ]; then
    echo "  Starting frontend..."
    cd "$SCRIPT_DIR/frontend"
    nohup npm run dev > /dev/null 2>&1 &
    cd "$SCRIPT_DIR"
    sleep 5
fi

echo ""
echo "  Waiting for services to start..."
sleep 8

echo ""
echo "  [OK] Services started!"
echo ""
echo "  Opening browser..."

if [[ "$OSTYPE" == "darwin"* ]]; then
    open "http://localhost:5173"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    xdg-open "http://localhost:5173" 2>/dev/null || echo "  Please open http://localhost:5173 in your browser"
fi

echo ""
echo "  Service URLs:"
echo "    Frontend:  http://localhost:5173"
echo "    Backend:   http://localhost:5001"
echo "    MaxKB:     http://localhost:8080"
echo ""

