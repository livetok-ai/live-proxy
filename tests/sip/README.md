# SIP Call CLI Tool

A simple command-line tool to initiate SIP calls using the [sipgo](https://github.com/emiago/sipgo) library.

## Requirements

- Go 1.21 or higher
- Python 3.12+ (for testing with sip.py server)

## Installation

Navigate to the `tests/sip` directory and build the binary:

```bash
cd tests/sip
./build.sh
```

Or manually:
```bash
cd tests/sip
go mod download
go build -o sipcli main.go
```

## Usage

Basic usage:

```bash
./sipcli -url sip:user@host:port
```

### Command-line Options

- `-url` (required): SIP URL to call (format: `sip:user@host:port`)
- `-username`: Username for SIP client (default: `sipcli`)
- `-duration`: Call duration in seconds, 0 for indefinite (default: `30`)
- `-user-agent`: User-Agent header (default: `SIPGoCLI/1.0`)
- `-display-name`: Display name for caller (default: `SIP CLI`)
- `-transport`: Transport protocol - `tcp` or `udp` (default: `tcp`)

### Examples

Make a 30-second call (default):
```bash
./sipcli -url sip:123456789@asterisk.example.com:5060
```

Make an indefinite call (hangup with Ctrl+C):
```bash
./sipcli -url sip:123456789@asterisk.example.com:5060 -duration 0
```

Make a 60-second call with custom user agent:
```bash
./sipcli -url sip:123456789@asterisk.example.com:5060 -duration 60 -user-agent "MyApp/2.0"
```

Full customization:
```bash
./sipcli \
  -url sip:123456789@asterisk.example.com:5060 \
  -username myclient \
  -duration 120 \
  -user-agent "CustomApp/1.0" \
  -display-name "John Doe"
```

## Features

- Simple CLI interface for making SIP calls
- Configurable call duration
- Graceful shutdown with Ctrl+C
- Custom SIP headers support
- Signal handling for clean hangup

## Testing with sip.py Server

The repository includes a Python-based SIP server ([sip.py](../../sip.py)) for testing. You can use the provided test script to run both the server and client together:

```bash
./test.sh
```

This will:
1. Start the Python SIP server on port 5060
2. Run the Go SIP client to connect to localhost:5060
3. Automatically cleanup when done

### Manual Testing

Start the Python SIP server manually:
```bash
# From the repository root
python3 sip.py
```

In another terminal, run the Go client:
```bash
cd tests/sip
./sipcli -url sip:testuser@localhost:5060 -transport tcp -duration 5
```

The SIP server will log received messages, allowing you to verify the client is working correctly.

## Development

Run the tool without building:
```bash
go run main.go -url sip:user@host:port
```

## Notes

- The tool automatically adds an `X-SIP-CLI` header to identify calls made by this CLI
- Press Ctrl+C at any time to gracefully hangup the call
- When duration is set to 0, the call will continue until manually interrupted
- TCP transport is used by default for compatibility with the Python sip.py server
- UDP transport is also supported via the `-transport udp` flag
