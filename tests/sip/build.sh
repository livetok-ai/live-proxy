#!/bin/bash
# Build script for SIP CLI tool

set -e

echo "Building SIP CLI tool..."

# Check if Go is installed
if ! command -v go &> /dev/null; then
    echo "Error: Go is not installed"
    echo "Please install Go 1.21 or higher from https://golang.org/dl/"
    exit 1
fi

# Download dependencies and update go.sum
echo "Downloading dependencies..."
go mod tidy
go mod download

# Build the binary
echo "Building binary..."
go build -o sipcli main.go

echo "Build successful! Binary created: sipcli"
echo ""
echo "Usage: ./sipcli -url sip:user@host:port"
