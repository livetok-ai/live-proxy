**Live-Proxy is a [LiveTok Labs project](https://www.livetok.io)**

# Live-Proxy

Live-Proxy is an open-source proxy that bridges real-time audio/video protocols — **WebRTC**, **SIP**, **RTMP**, and **WebTransport/raw QUIC** — to AI models exposed over WebSocket, streaming, or request/response APIs (LLMs, speech, vision, and simulation models). It lets a phone call, an RTMP stream from OBS/ffmpeg, or a browser peer connection talk directly to a model without the client having to speak that model's native API, and it exposes an **HTTP control interface** to create, update, and tear down those sessions programmatically.

## How it works

- **Interfaces** (`interfaces/`) accept media from the outside world and normalize it into a connection: `webrtc` (via `connection.py`), `sip`, `rtmp`, and `webtransport` (HTTP/3 WebTransport + raw QUIC).
- **Providers** (`providers/`) wrap each AI model behind a common `Model` interface (`model.py`) with `send`/`recv` for streaming audio, video, and text, plus event hooks for transcriptions, interruptions, and detections.
- **The HTTP API** (`proxy.py`) lets a backend service create a connection (choosing the interface and model), update it mid-call, list active sessions, and tear it down — see [HTTP control interface](#http-control-interface) below.

## Getting Started

### Prerequisites

Python 3.12+ is required. [uv](https://docs.astral.sh/uv/) is recommended for managing the virtual environment and dependencies.

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

1. Set the API key(s) for the model(s) you plan to use and run the server:
   ```bash
   # For Gemini (LLM, robotics, TTS)
   GOOGLE_API_KEY=your_key_here python proxy.py

   # For OpenAI (LLM)
   OPENAI_API_KEY=your_key_here python proxy.py

   # For Cartesia (TTS)
   CARTESIA_API_KEY=your_key_here python proxy.py
   ```
   Local/self-hosted models (YOLO, SAM3, OCR, MediaPipe face landmarker, sentiment, local LLM, MuJoCo, Insivision) don't need an API key — see [Models supported](#models-supported).

   Useful flags: `--host`/`--port` (HTTP + WebRTC signaling, default `0.0.0.0:8080`), `--sip-port` (default `5060`), `--rtmp-port` (default `1935`), `--wt-port` (default `4433`), `--log-level`. Run `python proxy.py --help` for the full list.

2. Open the demo page in your browser:
   ```
   http://localhost:8080/demo/index.html
   ```

3. Click **Start** and begin talking with the model!

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

### Running the Test Suite

```bash
uv run pytest              # full suite
uv run pytest --cov=.      # with coverage
uv run pytest tests/test_proxy.py -v   # a single file, verbose
```

See [README_DEV.md](README_DEV.md) for formatting/linting/type-checking commands.

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
    model: 'gemini' // see "Models supported" for other values
  })
});

const answer = await response.json();
await pc.setRemoteDescription({ type: 'answer', sdp: answer.sdp });
```

### SIP Integration

Point a SIP client/PBX at the proxy's SIP port (default `5060`) and dial in; the call is bridged to the model selected via the SIP session setup (see `interfaces/sip/`). Use `--sip-host`/`--sip-port` to change the listening address, and `--sip-callback-url` (or `SIP_CALLBACK_URL`) to receive call lifecycle events on your backend.

### RTMP Integration

Publish an RTMP stream (e.g. from OBS or ffmpeg) to `rtmp://host:1935/live/<key>?model=<model>`, where the query string on the stream key carries the same connection parameters as the `/connection` HTTP body (`model`, `system_instructions`, `voice`, `api_key`, ...). Use `--rtmp-host`/`--rtmp-port` to change the listening address, and `--rtmp-callback-url` (or `RTMP_CALLBACK_URL`) to receive session events on your backend.

```bash
ffmpeg -re -i input.mp4 -c copy -f flv "rtmp://localhost:1935/live/mystream?model=gemini"
```

### WebTransport / raw QUIC Integration

The proxy also runs a QUIC server (default UDP port `4433`, override with `--wt-port`) that accepts both
browser [WebTransport](https://developer.mozilla.org/en-US/docs/Web/API/WebTransport) sessions (over HTTP/3)
and raw QUIC clients on a `live-proxy-quic` ALPN, without needing an SDP offer/answer exchange.

A client opens a single bidirectional stream and writes length-prefixed binary frames
(`[4B length][1B type][8B timestamp_us][type-specific extra][payload]`, see `interfaces/webtransport/protocol.py`):
type `1` for audio (raw PCM s16le, extra = sample rate + channel count), type `2` for video (JPEG, extra = keyframe
flag), and type `3` for control messages (UTF-8 JSON). The very first control frame must carry the connection
parameters (`model`, `system_instructions`, `tools`, `voice`, `language`, `api_key`, `metadata`, ...) — the same
fields as the `/connection` HTTP body, minus `sdp`.

Try it live at `http://localhost:8080/demo/webtransport.html` (works with the server's own ephemeral, short-lived
self-signed certificate — the page fetches its SHA-256 hash from `GET /webtransport-info` and pins it via
`serverCertificateHashes`). Pass `--wt-cert-file`/`--wt-key-file` to use a real certificate instead. Raw QUIC can't
be driven from a browser (no raw socket API); see `tests/test_webtransport.py` for a scripted client example.

## HTTP control interface

Every session, regardless of which media interface it came in on, can be managed over plain HTTP from a backend service:

| Method | Path | Description |
|---|---|---|
| `POST` | `/connection` | Create a connection (WebRTC offer/answer exchange plus model selection and parameters). |
| `PUT` | `/connection/{connection_id}` | Update a live connection (e.g. change system instructions, voice, enabled tools). |
| `DELETE` | `/connection/{connection_id}` | Tear down a connection. |
| `GET` | `/session` | List active sessions. |
| `POST` | `/session/{session_id}/connection` | Create a connection within an existing session. |
| `GET` | `/metrics` | Prometheus metrics. |
| `GET` | `/webtransport-info` | Fetch the server's ephemeral WebTransport certificate hash for pinning. |

The same connection parameters (`model`, `system_instructions`, `tools`, `voice`, `language`, `api_key`, `metadata`, ...) are shared across the HTTP, RTMP, and WebTransport entry points, so switching a client from one transport to another doesn't change how a session is configured.

## Models supported

The `model` parameter (or `model` query param for RTMP) selects the provider; prefixes are matched against `MODEL_MAP` in `connection.py`, so e.g. `gemini-robotics` resolves ahead of `gemini`.

| `model` value | Provider | Kind | Notes |
|---|---|---|---|
| `gemini` | Google Gemini | Speech-to-speech / multimodal LLM | Requires `GOOGLE_API_KEY` (or Vertex AI via `GOOGLE_GENAI_USE_VERTEXAI` + service account). |
| `openai` | OpenAI Realtime | Speech-to-speech LLM | Requires `OPENAI_API_KEY`. |
| `gemini-robotics` | Gemini Robotics-ER | Vision — object detection for robot manipulation | Requires `GOOGLE_API_KEY`. |
| `cosmos` | NVIDIA Cosmos (`nvidia/Cosmos3-Nano` by default) | Vision — sliding-window video understanding, run locally via `transformers` | Downloaded and cached on first use; no API key. Needs a GPU for reasonable latency. |
| `local_llm` | Hugging Face `transformers` | Local/self-hosted text LLM | Downloaded and cached on first use; no API key. |
| `cartesia` | Cartesia | Text-to-speech | Requires `CARTESIA_API_KEY`. |
| `simli` | Simli | Real-time avatar video (audio-driven talking face) | Requires `SIMLI_API_KEY` (+ `SIMLI_FACE_ID`). |
| `yolo` | Ultralytics YOLO | Vision — real-time object detection | Local model, no API key. |
| `sam3` / `sam2` / `sam` | Ultralytics SAM | Vision — segmentation | Local model, no API key. |
| `ocr` | EasyOCR | Vision — text recognition | Local model, no API key. |
| `face_landmarker` | MediaPipe Face Landmarker | Vision — facial landmark detection | Model weights fetched from Google's model store on first use. |
| `inception` | FaceNet (Inception) | Vision — face embeddings | Local model, no API key. |
| `text_sentiment` | Sentiment classifier | Text — sentiment analysis | Local model, no API key. |
| `insivision` | Insivision | Vision — proxies frames to an external Insivision WebSocket server | Requires `INSIVISION_URL` (default `ws://localhost:8766`). |
| `mujoco` | MuJoCo | In-process physics simulation rendered as video | Requires `MUJOCO_SCENE_XML`. |

Each provider declares which of `supports_audio`, `supports_video`, and `supports_text` it implements, so the proxy only wires up the media tracks a given model actually accepts.

## Features

- ✅ **WebRTC interface** - Real-time peer-to-peer communication
- ✅ **SIP interface** - Session Initiation Protocol support for telephony integration
- ✅ **RTMP interface** - Ingest from RTMP publishers (OBS, ffmpeg, ...)
- ✅ **WebTransport interface** - HTTP/3 WebTransport and raw QUIC support
- ✅ **HTTP control interface** - Create, update, list, and tear down sessions from a backend service
- ✅ **Audio support** - Bidirectional audio streaming
- ✅ **Video support** - Video streaming capabilities
- ✅ **Data channels** - Send and receive messages alongside media
- ✅ **Multiple AI providers** - LLM, speech, vision, and simulation models (see [Models supported](#models-supported))

## Documentation

- [Development Setup](README_DEV.md) - Setup instructions for contributors
- [Certificate Setup](CERT.md) - SSL certificate configuration guide

## License

See [LICENSE](LICENSE) file for details.
