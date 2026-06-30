**Live-Proxy is a [LiveTok Labs project](https://www.livetok.io)**

# Live-Proxy

Live-Proxy is an open-source proxy service for interacting with large language model (LLM) WebSocket APIs, exposing interfaces better suited for real-time communication over the Internet. It supports Gemini and OpenAI speech-to-speech models and facilitates real-time communication using WebRTC, SIP, and WebTransport.

## Getting Started

### Prerequisites

Python 3.8+ is required. [uv](https://docs.astral.sh/uv/) is recommended for managing the virtual environment and dependencies.

### Setup with uv

```bash
# Create a virtual environment
uv venv

# Activate it
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate       # Windows

# Install dependencies
uv sync
```

After activating the venv you can run scripts directly with `python`, or skip activation and use `uv run python` instead (uv resolves the environment automatically).

See [README_DEV.md](README_DEV.md) for detailed development setup instructions.

### Running the Server

1. Set your API key environment variable and run the server:
   ```bash
   # For Gemini
   GOOGLE_API_KEY=your_key_here python proxy.py
   
   # For OpenAI
   OPENAI_API_KEY=your_key_here python proxy.py
   ```

2. Open the demo page in your browser:
   ```
   http://localhost:8080/demo/index.html
   ```

3. Click **Start** and begin talking with the LLM!

### Running the RTC Test Client

We provide a Python-based RTC test client that streams a sample MP4 video (including audio and video tracks) to the proxy server to verify real-time speech-to-text transcriptions with the Gemini Multimodal Live API.

1. Ensure your API key is configured in `live-proxy/.env` (e.g. `GOOGLE_API_KEY=your_key`).
2. Start the proxy server:
   ```bash
   uv run python proxy.py --port 8081
   ```
3. In a separate terminal, run the test client:
   * **If inside the `live-proxy` directory:**
     ```bash
     uv run python tests/rtc_test_client.py
     ```
   * **From the repository root:**
     ```bash
     uv run --directory live-proxy python tests/rtc_test_client.py
     ```
4. Observe the proxy server logs to see the real-time transcriptions returned by the Gemini model!

## Using it in Your Own Application

### WebRTC Integration

```js
// Get user's audio stream
const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

// Create peer connection
const pc = new RTCPeerConnection();

// Handle incoming audio track
pc.ontrack = e => audioElement.srcObject = e.streams[0];

// Add local audio track
pc.addTrack(stream.getTracks()[0]);

// Create and send offer
const offer = await pc.createOffer();
await pc.setLocalDescription(offer);

// Send offer to Live-Proxy and receive answer
const response = await fetch('http://localhost:8080/connection', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    sdp: offer.sdp,
    type: offer.type,
    model: 'gemini' // or 'openai'
  })
});

const answer = await response.json();
await pc.setRemoteDescription({ type: 'answer', sdp: answer.sdp });
```

## Features

- ✅ **WebRTC interface** - Real-time peer-to-peer communication
- ✅ **SIP interface** - Session Initiation Protocol support for telephony integration
- ✅ **Audio support** - Bidirectional audio streaming
- ✅ **Video support** - Video streaming capabilities
- ✅ **Data channels** - Send and receive messages alongside media
- ✅ **Gemini integration** - Google's Gemini 2.0 multimodal models
- ✅ **OpenAI integration** - OpenAI's GPT-4 real-time API
- 🚧 **WebTransport interface** - Work in progress
- 📋 **HTTP real-time control interface** - Planned

## Documentation

- [Development Setup](README_DEV.md) - Setup instructions for contributors
- [Certificate Setup](CERT.md) - SSL certificate configuration guide

## License

See [LICENSE](LICENSE) file for details.
