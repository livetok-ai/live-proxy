#!/bin/bash
# Test script to run SIP server and client together

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}Starting SIP integration test${NC}"
echo ""

# Check if Go binary exists
if [ ! -f "sipcli" ]; then
    echo -e "${RED}Error: sipcli binary not found${NC}"
    echo "Please run ./build.sh first"
    exit 1
fi

# Check if sip.py exists
if [ ! -f "../../sip.py" ]; then
    echo -e "${RED}Error: sip.py not found at ../../sip.py${NC}"
    exit 1
fi

# Start Python SIP server in background
echo -e "${GREEN}Starting Python SIP server on port 5060...${NC}"
(cd ../.. && SIP_CALLBACK_URL=https://api.livetok.io/api/live-proxy/sip uv run python3 sip.py) &
SIP_SERVER_PID=$!

# Give server time to start
sleep 2

# Check if server is running
if ! ps -p $SIP_SERVER_PID > /dev/null; then
    echo -e "${RED}Error: SIP server failed to start${NC}"
    exit 1
fi

echo -e "${GREEN}SIP server started with PID: $SIP_SERVER_PID${NC}"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo -e "${BLUE}Cleaning up...${NC}"
    if ps -p $SIP_SERVER_PID > /dev/null; then
        echo "Stopping SIP server (PID: $SIP_SERVER_PID)"
        kill $SIP_SERVER_PID 2>/dev/null || true
        wait $SIP_SERVER_PID 2>/dev/null || true
    fi
    echo -e "${GREEN}Done${NC}"
}

trap cleanup EXIT INT TERM

# Run the Go SIP client
echo -e "${GREEN}Running Go SIP client...${NC}"
echo -e "${BLUE}Command: ./sipcli -url sip:33@localhost:5060 -transport tcp -username 123 -duration 5${NC}"
echo ""

./sipcli -url sip:33@localhost:5060 -transport tcp -duration 5

echo ""
echo -e "${GREEN}Test completed successfully!${NC}"
