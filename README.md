**Live-Proxy is a [LiveTok Labs project](https://www.livetok.io)**

# Live-Proxy

Live-Proxy is an open-source proxy service for interacting with large language model (LLM) WebSocket APIs, exposing interfaces better suited for real-time communication over the Internet. It supports Gemini and OpenAI speech-to-speech models and facilitates real-time communication using WebRTC, SIP, and WebTransport.

## Getting Started

### Prerequisites

Python 3.8+ is required. See [README_DEV.md](README_DEV.md) for detailed development setup instructions.

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
