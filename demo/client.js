// get DOM elements
const dataChannelLog = document.getElementById('data-channel'),
  iceConnectionLog = document.getElementById('ice-connection-state');

var pc = null;
var dc = null;        // reliable datachannel (transcriptions, control messages)
var dcUnreliable = null; // unreliable datachannel (key events)
var heartbeatInterval = null;
var lastMessageElement = null; // Track the last message bubble element
var isAppendingToLast = false; // Track if we're appending to the last message

// Which transport is currently connected: 'webrtc' | 'websocket' | 'webtransport' | null
var activeTransport = null;

// WebSocket transport state
var socket = null;

// WebTransport transport state
var wtTransport = null;
var wtWriter = null;
var wtReader = null;

// Shared media capture/playback state (websocket & webtransport transports)
var mediaStream = null;
var streamAudioContext = null;
var playbackContext = null;
var nextPlaybackTime = 0;
var videoSendInterval = null;

// Read from query params
const urlParams = new URLSearchParams(window.location.search);
const BASE_URL = urlParams.get('base_url') || '';

// --- Transport selection helpers -------------------------------------------

function getTransport() {
  const el = document.getElementById('transport');
  return (el && el.value) || 'webrtc';
}

function disableTransportSelect(disabled) {
  const el = document.getElementById('transport');
  if (el) el.disabled = disabled;
}

function resolveHost() {
  try {
    return new URL(BASE_URL || window.location.href, window.location.href).hostname;
  } catch (e) {
    return window.location.hostname;
  }
}

function setConnectionStatus(text) {
  if (iceConnectionLog) iceConnectionLog.textContent = text;
}

function setVideoDisplayMode(transport) {
  const videoEl = document.getElementById('video');
  const canvasEl = document.getElementById('remote-canvas');
  if (videoEl) videoEl.classList.toggle('hidden', transport !== 'webrtc');
  if (canvasEl) canvasEl.classList.toggle('hidden', transport === 'webrtc');
}

function resetConversationLog() {
  lastMessageElement = null;
  isAppendingToLast = false;
  if (dataChannelLog) dataChannelLog.innerHTML = '';
}

// --- Binary frame protocol (mirrors interfaces/webtransport/protocol.py) --
// Shared by the WebSocket and WebTransport transports: [4B length][1B type][8B timestamp_us][extra][payload]

const FRAME_TYPE_AUDIO = 1;
const FRAME_TYPE_VIDEO = 2;
const FRAME_TYPE_CONTROL = 3;

const STREAMING_AUDIO_SAMPLE_RATE = 16000;
const STREAMING_VIDEO_FPS = 8;

function packFrame(type, timestampUs, extra, payload) {
  const header = new Uint8Array(1 + 8 + extra.length);
  header[0] = type;
  // 8-byte big-endian timestamp (safe for the range we use here)
  const hi = Math.floor(timestampUs / 2 ** 32);
  const lo = timestampUs >>> 0;
  new DataView(header.buffer).setUint32(1, hi);
  new DataView(header.buffer).setUint32(5, lo);
  header.set(extra, 9);

  const body = new Uint8Array(header.length + payload.length);
  body.set(header, 0);
  body.set(payload, header.length);

  const out = new Uint8Array(4 + body.length);
  new DataView(out.buffer).setUint32(0, body.length);
  out.set(body, 4);
  return out;
}

function packAudio(pcm16, sampleRate, channels, timestampUs) {
  const extra = new Uint8Array(5);
  const dv = new DataView(extra.buffer);
  dv.setUint32(0, sampleRate);
  dv.setUint8(4, channels);
  return packFrame(FRAME_TYPE_AUDIO, timestampUs, extra, new Uint8Array(pcm16.buffer, pcm16.byteOffset, pcm16.byteLength));
}

function packVideo(jpegBytes, timestampUs, keyframe = true) {
  const extra = new Uint8Array([keyframe ? 1 : 0]);
  return packFrame(FRAME_TYPE_VIDEO, timestampUs, extra, jpegBytes);
}

function packControl(jsonObj) {
  const payload = new TextEncoder().encode(JSON.stringify(jsonObj));
  return packFrame(FRAME_TYPE_CONTROL, 0, new Uint8Array(0), payload);
}

class FrameDecoder {
  constructor() {
    this._buffer = new Uint8Array(0);
  }

  feed(chunk) {
    const merged = new Uint8Array(this._buffer.length + chunk.length);
    merged.set(this._buffer, 0);
    merged.set(chunk, this._buffer.length);
    this._buffer = merged;

    const frames = [];
    while (true) {
      if (this._buffer.length < 4) break;
      const length = new DataView(this._buffer.buffer, this._buffer.byteOffset).getUint32(0);
      if (this._buffer.length < 4 + length) break;
      const body = this._buffer.slice(4, 4 + length);
      this._buffer = this._buffer.slice(4 + length);
      frames.push(this._parse(body));
    }
    return frames;
  }

  _parse(body) {
    const dv = new DataView(body.buffer, body.byteOffset);
    const type = dv.getUint8(0);
    const timestampUs = Number(dv.getBigUint64(1));
    let offset = 9;
    let extra = {};
    if (type === FRAME_TYPE_AUDIO) {
      extra = { sampleRate: dv.getUint32(offset), channels: dv.getUint8(offset + 4) };
      offset += 5;
    } else if (type === FRAME_TYPE_VIDEO) {
      extra = { keyframe: !!(dv.getUint8(offset) & 1) };
      offset += 1;
    }
    return { type, timestampUs, extra, payload: body.slice(offset) };
  }
}

function base64ToBytes(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes;
}

// Function to add or append message to chat bubbles
function addOrAppendMessage(messageType, content) {
  const isUser = messageType === 'user';
  const lastWasSameType = lastMessageElement && lastMessageElement.dataset.role === messageType;

  if (lastWasSameType) {
    // Append to existing message bubble
    const messageText = lastMessageElement.querySelector('.message-text');
    messageText.textContent += content;
  } else {
    // Create new message bubble
    const messageDiv = document.createElement('div');
    messageDiv.dataset.role = messageType;

    if (isUser) {
      messageDiv.className = 'flex justify-end';
      const bubble = document.createElement('div');
      bubble.className = 'max-w-xs lg:max-w-md px-4 py-2 rounded-lg bg-primary text-white break-words word-wrap';
      const textDiv = document.createElement('div');
      textDiv.className = 'text-sm';
      const textSpan = document.createElement('span');
      textSpan.className = 'message-text';
      textSpan.textContent = content;
      textDiv.appendChild(textSpan);
      bubble.appendChild(textDiv);
      messageDiv.appendChild(bubble);
    } else {
      messageDiv.className = 'flex justify-start';
      const bubble = document.createElement('div');
      bubble.className = 'max-w-xs lg:max-w-md px-4 py-2 rounded-lg bg-white border shadow-sm break-words word-wrap';
      const textDiv = document.createElement('div');
      textDiv.className = 'text-sm text-gray-800';
      const textSpan = document.createElement('span');
      textSpan.className = 'message-text';
      textSpan.textContent = content;
      textDiv.appendChild(textSpan);
      bubble.appendChild(textDiv);
      messageDiv.appendChild(bubble);
    }

    dataChannelLog.appendChild(messageDiv);
    lastMessageElement = messageDiv;
  }

  // Auto-scroll to bottom
  dataChannelLog.scrollTop = dataChannelLog.scrollHeight;
}

// Show the latest per-frame inference result, replacing whatever was shown before.
function updateInferenceStatus(provider, content) {
  const el = document.getElementById('inference-status');
  if (!el) return;

  let formatted;
  try {
    formatted = JSON.stringify(content, null, 2);
  } catch (e) {
    formatted = String(content);
  }

  el.textContent = `[${provider}] ${formatted}`;
}

// Deterministic color assignment for labels
const detectionColors = [
  'rgb(255, 75, 75)',    // Red
  'rgb(75, 123, 255)',   // Blue
  'rgb(75, 255, 123)',   // Green
  'rgb(180, 75, 255)',   // Purple
  'rgb(255, 140, 0)',    // Orange
  'rgb(0, 206, 209)',    // Cyan
  'rgb(255, 215, 0)',    // Yellow
  'rgb(255, 105, 180)',  // Pink
  'rgb(255, 20, 147)',   // Deep Pink
  'rgb(0, 250, 154)'     // Medium Spring Green
];

function getColorForLabel(label) {
  let h = 0;
  for (let i = 0; i < label.length; i++) {
    h = (h * 31 + label.charCodeAt(i)) & 0xFFFFFFFF;
  }
  return detectionColors[Math.abs(h) % detectionColors.length];
}

// Log every non-heartbeat data channel message (sent or received) to the
// console, so the datachannel traffic can be inspected while debugging.
function isHeartbeatMessage(data) {
  return data === '{}';
}

function logDataChannelMessage(direction, channelLabel, data) {
  if (isHeartbeatMessage(data)) return;
  console.log(`[datachannel:${channelLabel}] ${direction}`, data);
}

function sendDataChannelMessage(channel, data) {
  logDataChannelMessage('send', channel && channel.label, data);
  channel.send(data);
}

function renderObjects(objects) {
  const container = document.getElementById('detection-boxes-container');
  const labelsContainer = document.getElementById('detection-labels-container');

  if (container) container.innerHTML = '';
  if (labelsContainer) labelsContainer.innerHTML = '';

  if (!objects || !Array.isArray(objects)) return;

  objects.forEach(obj => {
    const { label, top, left, bottom, right } = obj;
    const hasCoords = top !== undefined && left !== undefined && bottom !== undefined && right !== undefined;

    if (!hasCoords) {
      // No bounding box for this detection: surface the label as a pill
      // overlay at the top of the video instead of drawing a box.
      if (!labelsContainer) return;

      const pill = document.createElement('span');
      pill.style.backgroundColor = 'rgba(0, 0, 0, 0.5)';
      pill.style.border = '1px solid white';
      pill.style.color = 'white';
      pill.style.fontSize = '12px';
      pill.style.fontWeight = 'bold';
      pill.style.padding = '4px 10px';
      pill.style.borderRadius = '10px';
      pill.style.whiteSpace = 'normal';
      pill.style.wordBreak = 'break-word';
      pill.style.maxWidth = '100%';
      pill.textContent = label;

      labelsContainer.appendChild(pill);
      return;
    }

    if (!container) return;

    // Convert relative coordinates [0, 1] to percentages
    const pctLeft = (left * 100).toFixed(2) + '%';
    const pctTop = (top * 100).toFixed(2) + '%';
    const pctWidth = ((right - left) * 100).toFixed(2) + '%';
    const pctHeight = ((bottom - top) * 100).toFixed(2) + '%';

    const color = getColorForLabel(label);

    const box = document.createElement('div');
    box.style.position = 'absolute';
    box.style.left = pctLeft;
    box.style.top = pctTop;
    box.style.width = pctWidth;
    box.style.height = pctHeight;
    box.style.border = `3px solid ${color}`;
    box.style.pointerEvents = 'none';

    // Label tag
    const labelSpan = document.createElement('span');
    labelSpan.style.position = 'absolute';
    labelSpan.style.top = '-22px';
    labelSpan.style.left = '-3px';
    labelSpan.style.backgroundColor = color;
    labelSpan.style.color = 'white';
    labelSpan.style.fontSize = '12px';
    labelSpan.style.fontWeight = 'bold';
    labelSpan.style.padding = '2px 6px';
    labelSpan.style.whiteSpace = 'nowrap';
    labelSpan.textContent = label;

    box.appendChild(labelSpan);
    container.appendChild(box);
  });
}

// Handle a parsed control/data-channel JSON payload. This is the single
// place that reacts to server messages, shared across all three transports.
function processControlPayload(data) {
  if (!data || typeof data !== 'object') return;

  // Show on top of the video feed any data received that has a "display" attribute
  if ('display' in data) {
    const overlay = document.getElementById('video-overlay');
    if (overlay) {
      if (data.display !== null && data.display !== undefined && String(data.display).trim() !== '') {
        overlay.textContent = data.display;
        overlay.classList.remove('hidden');
      } else {
        overlay.textContent = '';
        overlay.classList.add('hidden');
      }
    }
  }

  if (data.type === 'transcription') {
    addOrAppendMessage(data.role, data.content);
  } else if (data.type === 'inference') {
    // Latest raw inference result for a processed frame — replaces
    // whatever was previously shown, it is not appended.
    updateInferenceStatus(data.provider, data.content);
  } else if (data.type === 'objects') {
    renderObjects(data.objects);
  }
}

function handleDataChannelMessage(evt) {
  const message = evt.data;

  logDataChannelMessage('recv', evt.target && evt.target.label, message);

  try {
    // Try to parse as JSON
    const data = JSON.parse(message);
    processControlPayload(data);
  } catch (e) {
    // Fallback for non-JSON messages (backwards compatibility)
    const currentMessageType = message.startsWith('<') ? 'model' : message.startsWith('>') ? 'user' : null;

    if (currentMessageType) {
      // Extract the actual message content (without < or > prefix)
      const content = message.substring(2);
      addOrAppendMessage(currentMessageType, content);
    }
    // Ignore other non-JSON messages
  }
}

function handleControlFrame(frame) {
  try {
    const data = JSON.parse(new TextDecoder().decode(frame.payload));
    processControlPayload(data);
  } catch (e) {
    console.log('[control] non-JSON payload:', new TextDecoder().decode(frame.payload));
  }
}

function handleIncomingFrame(frame) {
  if (frame.type === FRAME_TYPE_AUDIO) playAudioFrame(frame);
  else if (frame.type === FRAME_TYPE_VIDEO) showVideoFrame(frame);
  else if (frame.type === FRAME_TYPE_CONTROL) handleControlFrame(frame);
}

function playAudioFrame(frame) {
  if (!playbackContext) {
    playbackContext = new AudioContext();
    nextPlaybackTime = playbackContext.currentTime;
  }
  const sampleRate = frame.extra.sampleRate || 48000;
  const pcm16 = new Int16Array(frame.payload.buffer, frame.payload.byteOffset, frame.payload.length / 2);
  const buffer = playbackContext.createBuffer(1, pcm16.length, sampleRate);
  const channel = buffer.getChannelData(0);
  for (let i = 0; i < pcm16.length; i++) channel[i] = pcm16[i] / 32768;

  const src = playbackContext.createBufferSource();
  src.buffer = buffer;
  src.connect(playbackContext.destination);
  nextPlaybackTime = Math.max(nextPlaybackTime, playbackContext.currentTime);
  src.start(nextPlaybackTime);
  nextPlaybackTime += buffer.duration;
}

async function showVideoFrame(frame) {
  const canvas = document.getElementById('remote-canvas');
  if (!canvas) return;

  const blob = new Blob([frame.payload], { type: 'image/jpeg' });
  const bitmap = await createImageBitmap(blob);
  if (canvas.width !== bitmap.width || canvas.height !== bitmap.height) {
    canvas.width = bitmap.width;
    canvas.height = bitmap.height;
  }
  canvas.getContext('2d').drawImage(bitmap, 0, 0);
  bitmap.close();

  const mediaDiv = document.getElementById('media');
  if (mediaDiv) mediaDiv.style.display = 'block';
}

// Generic "send a control message to the server" used for both the
// WebRTC unreliable data channel and the WebSocket/WebTransport control frame.
function sendControlMessage(obj) {
  if (activeTransport === 'webrtc') {
    const ch = dcUnreliable && dcUnreliable.readyState === 'open' ? dcUnreliable : dc;
    if (!ch || ch.readyState !== 'open') return;
    sendDataChannelMessage(ch, JSON.stringify(obj));
  } else if (activeTransport === 'websocket' || activeTransport === 'webtransport') {
    sendFrame(packControl(obj)).catch(() => { });
  }
}

// Send raw framed bytes over whichever streaming transport (websocket/webtransport) is active.
async function sendFrame(bytes) {
  if (activeTransport === 'websocket') {
    if (socket && socket.readyState === WebSocket.OPEN) socket.send(bytes);
  } else if (activeTransport === 'webtransport') {
    if (wtWriter) await wtWriter.write(bytes);
  }
}

function createPeerConnection() {
  var config = {
  };

  pc = new RTCPeerConnection(config);

  pc.addEventListener('iceconnectionstatechange', () => {
    if (iceConnectionLog) iceConnectionLog.textContent += ' -> ' + pc.iceConnectionState;
  }, false);
  if (iceConnectionLog) iceConnectionLog.textContent = pc.iceConnectionState;

  // connect audio / video
  pc.addEventListener('track', (evt) => {
    const recvVideo = document.getElementById('recv-video');
    if (evt.track.kind == 'video' && (!recvVideo || recvVideo.checked)) {
      // Reduce jitter buffer to 100 ms for lower latency video playback
      if (evt.receiver && 'jitterBufferTarget' in evt.receiver) {
        evt.receiver.jitterBufferTarget = 0.1;
      }
      const videoEl = document.getElementById('video');
      const currentTrack = videoEl?.srcObject?.getVideoTracks?.()[0];
      if (videoEl && currentTrack?.id !== evt.track.id) videoEl.srcObject = evt.streams[0];
    } else {
      const audioEl = document.getElementById('audio');
      const currentTrack = audioEl?.srcObject?.getAudioTracks?.()[0];
      if (audioEl && currentTrack?.id !== evt.track.id) audioEl.srcObject = evt.streams[0];
    }
  });

  return pc;
}

function enumerateInputDevices() {
  const populateSelect = (select, devices) => {
    if (!select) return;
    let counter = 1;
    devices.forEach((device) => {
      const option = document.createElement('option');
      option.value = device.deviceId;
      option.text = device.label || ('Device #' + counter);
      select.appendChild(option);
      counter += 1;
    });
  };

  navigator.mediaDevices.enumerateDevices().then((devices) => {
    populateSelect(
      document.getElementById('audio-input'),
      devices.filter((device) => device.kind == 'audioinput')
    );
    populateSelect(
      document.getElementById('video-input'),
      devices.filter((device) => device.kind == 'videoinput')
    );
  }).catch((e) => {
    alert(e);
  });
}

// Build the session/model configuration shared by all three transports
// (WebRTC sends it alongside the SDP offer; WebSocket/WebTransport send it
// as the first control frame).
function buildSessionParams() {
  const modelValue = document.getElementById('model').value;
  let model = modelValue === 'none' ? '' : modelValue;

  const appendAddon = (addon) => {
    model = model ? model + ';' + addon : addon;
  };

  const sampling = Math.max(1, parseInt(document.getElementById('video-sampling')?.value, 10) || 10);

  if (document.getElementById('provider-simli')?.checked) {
    appendAddon('simli');
  }
  if (document.getElementById('provider-yolo')?.checked) {
    appendAddon(`yolo[sampling=${sampling}]`);
  }
  if (document.getElementById('provider-face-landmarker')?.checked) {
    appendAddon('face_landmarker');
  }
  if (document.getElementById('provider-text-sentiment')?.checked) {
    appendAddon('text_sentiment');
  }
  if (document.getElementById('provider-sam2')?.checked) {
    appendAddon(`sam[version=sam2.1_t.pt,sampling=${sampling}]`);
  }
  if (document.getElementById('provider-sam3')?.checked) {
    appendAddon(`sam[version=sam2.1_t.pt,sampling=${sampling}]`);
  }
  if (document.getElementById('provider-inception')?.checked) {
    appendAddon('inception');
  }
  if (document.getElementById('provider-gemini-robotics')?.checked) {
    appendAddon(`gemini-robotics[sampling=${sampling}]`);
  }
  if (document.getElementById('provider-external-websocket')?.checked) {
    appendAddon('external_websocket');
  }
  if (document.getElementById('provider-mujoco')?.checked) {
    appendAddon('mujoco');
  }
  if (document.getElementById('provider-cosmos')?.checked) {
    appendAddon(`cosmos[sampling=${sampling}]`);
  }

  const systemInstructions = document.getElementById('system-instructions').value.trim();
  const tools = JSON.parse(document.getElementById('tools').value.trim());
  const voice = document.getElementById('voice').value;
  const language = document.getElementById('language').value;
  const apiKey = document.getElementById('api-key').value.trim();
  const ragCorpus = document.getElementById('rag-corpus').value.trim();
  const recording = document.getElementById('recording')?.checked || false;

  const params = {
    system_instructions: systemInstructions,
    tools: tools,
    voice: voice,
    language: language,
    model: model,
    metadata: {
      agent_id: '123',
      session_id: '456',
      phone_number_id: '789',
      from: '101',
    },
    recording: recording,
  };

  if (apiKey) params.api_key = apiKey;
  if (ragCorpus) params.rag_corpus = ragCorpus;

  return params;
}

async function negotiate() {
  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);

  const body = Object.assign({ sdp: offer.sdp }, buildSessionParams());

  let response;
  let result;

  try {
    response = await fetch(`${BASE_URL}/connection`, {
      body: JSON.stringify(body),
      headers: {
        'Content-Type': 'application/json'
      },
      method: 'POST'
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    result = await response.json();
  } catch (error) {
    console.error('Connection error:', error);

    // Show error popup with instructions
    const errorMessage = `Connection failed: ${error.message}\n\n` +
      `Please check the following:\n` +
      `• Make sure you have entered a valid API key\n` +
      `• Check your network connection`;

    alert(errorMessage);
    window.location.reload();
  }
  currentSessionId = result.sessionId || null;
  refreshSessions();

  const answer = new RTCSessionDescription({
    type: 'answer',
    sdp: result.sdp,
  });
  await pc.setRemoteDescription(answer);
}

// --- Media capture shared by the WebSocket & WebTransport transports -------
// (WebRTC captures/sends tracks directly through RTCPeerConnection instead.)

async function startStreamingMedia() {
  const useAudio = document.getElementById('use-audio')?.checked || false;
  const sendVideo = document.getElementById('send-video')?.checked || false;

  const constraints = { audio: false, video: false };

  if (useAudio) {
    const audioConstraints = {};
    const device = document.getElementById('audio-input')?.value;
    if (device) audioConstraints.deviceId = { exact: device };
    constraints.audio = Object.keys(audioConstraints).length ? audioConstraints : true;
  }

  if (sendVideo) {
    const videoConstraints = { width: { max: 320 }, height: { max: 240 } };
    const device = document.getElementById('video-input')?.value;
    if (device) videoConstraints.deviceId = { exact: device };
    constraints.video = videoConstraints;
  }

  if (!constraints.audio && !constraints.video) return;

  mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
  if (constraints.audio) startStreamingAudioCapture(mediaStream);
  if (constraints.video) startStreamingVideoCapture(mediaStream);
}

function startStreamingAudioCapture(stream) {
  streamAudioContext = new AudioContext();
  const source = streamAudioContext.createMediaStreamSource(stream);
  const processor = streamAudioContext.createScriptProcessor(2048, 1, 1);
  const ratio = STREAMING_AUDIO_SAMPLE_RATE / streamAudioContext.sampleRate;

  processor.onaudioprocess = (event) => {
    const input = event.inputBuffer.getChannelData(0);
    const outLength = Math.floor(input.length * ratio);
    const pcm16 = new Int16Array(outLength);
    for (let i = 0; i < outLength; i++) {
      const sample = input[Math.floor(i / ratio)] || 0;
      pcm16[i] = Math.max(-32768, Math.min(32767, Math.round(sample * 32767)));
    }
    sendFrame(packAudio(pcm16, STREAMING_AUDIO_SAMPLE_RATE, 1, Date.now() * 1000)).catch(() => { });
  };

  source.connect(processor);
  processor.connect(streamAudioContext.destination);
}

function startStreamingVideoCapture(stream) {
  const video = document.createElement('video');
  video.srcObject = stream;
  video.muted = true;
  video.play();

  const canvas = document.createElement('canvas');
  canvas.width = 320;
  canvas.height = 240;
  const ctx = canvas.getContext('2d');

  videoSendInterval = setInterval(() => {
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob(
      async (blob) => {
        if (!blob) return;
        const jpegBytes = new Uint8Array(await blob.arrayBuffer());
        sendFrame(packVideo(jpegBytes, Date.now() * 1000)).catch(() => { });
      },
      'image/jpeg',
      0.7
    );
  }, 1000 / STREAMING_VIDEO_FPS);
}

// --- Transport-specific connect implementations -----------------------------

async function startWebRTC() {
  activeTransport = 'webrtc';
  setVideoDisplayMode('webrtc');
  disableTransportSelect(true);

  pc = createPeerConnection();

  dc = pc.createDataChannel('data', { ordered: true });
  dc.addEventListener('close', () => { });
  dc.addEventListener('open', () => { });

  // Unreliable channel for low-latency key events (fire-and-forget)
  dcUnreliable = pc.createDataChannel('keys', { ordered: false, maxRetransmits: 0 });
  dcUnreliable.addEventListener('open', () => {
    heartbeatInterval = setInterval(() => {
      if (dcUnreliable && dcUnreliable.readyState === 'open') {
        try { sendDataChannelMessage(dcUnreliable, '{}'); } catch (_) { }
      }
    }, 20);
  });
  dc.addEventListener('message', handleDataChannelMessage);

  // Build media constraints.
  const useAudio = document.getElementById('use-audio')?.checked || false;
  const sendVideo = document.getElementById('send-video')?.checked || false;
  const recvVideo = document.getElementById('recv-video')?.checked || false;

  const constraints = {
    audio: false,
    video: false,
  };

  if (useAudio) {
    const audioConstraints = {};

    const device = document.getElementById('audio-input')?.value;
    if (device) {
      audioConstraints.deviceId = { exact: device };
    }

    constraints.audio = Object.keys(audioConstraints).length ? audioConstraints : true;
  }

  if (sendVideo) {
    const videoConstraints = { width: { max: 320 }, height: { max: 240 } };

    const device = document.getElementById('video-input')?.value;
    if (device) {
      videoConstraints.deviceId = { exact: device };
    }

    constraints.video = Object.keys(videoConstraints).length ? videoConstraints : true;
  }

  // Add transceiver for receiving video even if not sending
  if (recvVideo && !sendVideo) {
    const vt = pc.addTransceiver('video', { direction: 'recvonly' });
    // Prefer H264; strip VP8 so it isn't offered at all.
    const caps = RTCRtpReceiver.getCapabilities?.('video');
    if (caps) {
      const h264 = caps.codecs.filter(c => c.mimeType === 'video/H264');
      const rest = caps.codecs.filter(c => c.mimeType !== 'video/H264' && c.mimeType !== 'video/VP8');
      const rtx = caps.codecs.filter(c => c.mimeType === 'video/rtx');
      try { vt.setCodecPreferences([...h264, ...rest, ...rtx]); } catch (_) { }
    }
    const mediaDiv = document.getElementById('media');
    if (mediaDiv) mediaDiv.style.display = 'block';
  }

  if (constraints.audio || constraints.video) {
    mediaStream = await navigator.mediaDevices.getUserMedia(constraints)
    mediaStream.getTracks().forEach((track) => {
      pc.addTrack(track, mediaStream);
    });
    if (constraints.video) {
      const mediaDiv = document.getElementById('media');
      if (mediaDiv) mediaDiv.style.display = 'block';
      // Skip the local preview if remote video is expected too, since the
      // 'track' handler will immediately replace it, causing a flicker.
      if (!recvVideo) {
        const videoEl = document.getElementById('video');
        if (videoEl) videoEl.srcObject = mediaStream;
      }
    }
    await negotiate();
  } else {
    await negotiate();
  }

  const stopBtn = document.getElementById('stop');
  if (stopBtn) stopBtn.style.display = 'inline-block';
}

async function startWebSocket() {
  activeTransport = 'websocket';
  setVideoDisplayMode('websocket');
  disableTransportSelect(true);
  setConnectionStatus('connecting');

  const info = await (await fetch(`${BASE_URL}/websocket-info`)).json();
  const host = resolveHost();
  const scheme = info.secure ? 'wss' : 'ws';

  socket = new WebSocket(`${scheme}://${host}:${info.port}/`);
  socket.binaryType = 'arraybuffer';

  await new Promise((resolve, reject) => {
    socket.addEventListener('open', () => resolve(), { once: true });
    socket.addEventListener('error', () => reject(new Error('WebSocket connection failed')), { once: true });
  });
  setConnectionStatus('connected');

  const decoder = new FrameDecoder();
  socket.addEventListener('message', (event) => {
    for (const frame of decoder.feed(new Uint8Array(event.data))) handleIncomingFrame(frame);
  });
  socket.addEventListener('close', () => setConnectionStatus(iceConnectionLog.textContent + ' -> disconnected'));
  socket.addEventListener('error', () => setConnectionStatus(iceConnectionLog.textContent + ' -> failed'));

  await sendFrame(packControl(buildSessionParams()));
  await startStreamingMedia();

  const stopBtn = document.getElementById('stop');
  if (stopBtn) stopBtn.style.display = 'inline-block';
}

async function readWebTransportLoop() {
  const decoder = new FrameDecoder();
  while (activeTransport === 'webtransport') {
    const { value, done } = await wtReader.read();
    if (done) break;
    for (const frame of decoder.feed(value)) handleIncomingFrame(frame);
  }
}

async function startWebTransport() {
  activeTransport = 'webtransport';
  setVideoDisplayMode('webtransport');
  disableTransportSelect(true);
  setConnectionStatus('connecting');

  const info = await (await fetch(`${BASE_URL}/webtransport-info`)).json();
  const host = resolveHost();

  const options = {};
  if (info.certificateHash) {
    options.serverCertificateHashes = [{ algorithm: 'sha-256', value: base64ToBytes(info.certificateHash).buffer }];
  }

  wtTransport = new WebTransport(`https://${host}:${info.port}/webtransport`, options);
  wtTransport.closed
    .then(() => setConnectionStatus(iceConnectionLog.textContent + ' -> disconnected'))
    .catch(() => setConnectionStatus(iceConnectionLog.textContent + ' -> failed'));
  await wtTransport.ready;
  setConnectionStatus('connected');

  const stream = await wtTransport.createBidirectionalStream();
  wtWriter = stream.writable.getWriter();
  wtReader = stream.readable.getReader();
  readWebTransportLoop();

  await sendFrame(packControl(buildSessionParams()));
  await startStreamingMedia();

  const stopBtn = document.getElementById('stop');
  if (stopBtn) stopBtn.style.display = 'inline-block';
}

async function start() {
  const startBtn = document.getElementById('start');
  if (startBtn) startBtn.style.display = 'none';

  resetConversationLog();

  const transport = getTransport();
  if (transport === 'websocket') {
    await startWebSocket();
  } else if (transport === 'webtransport') {
    await startWebTransport();
  } else {
    await startWebRTC();
  }
}

async function stop() {
  const stopBtn = document.getElementById('stop');
  if (stopBtn) stopBtn.style.display = 'none';
  const startBtn = document.getElementById('start');
  if (startBtn) startBtn.style.display = 'inline-block';

  currentSessionId = null;
  refreshSessions();

  activeTransport = null;

  if (pc) {
    pc.close();
    pc = null;
  }
  dc = null;
  dcUnreliable = null;
  if (heartbeatInterval) { clearInterval(heartbeatInterval); heartbeatInterval = null; }

  if (socket) {
    try { socket.close(); } catch (_) { }
    socket = null;
  }

  if (videoSendInterval) { clearInterval(videoSendInterval); videoSendInterval = null; }
  if (streamAudioContext) { await streamAudioContext.close().catch(() => { }); streamAudioContext = null; }

  if (wtWriter) { try { await wtWriter.close(); } catch (_) { } wtWriter = null; }
  wtReader = null;
  if (wtTransport) { try { wtTransport.close(); } catch (_) { } wtTransport = null; }

  if (playbackContext) { await playbackContext.close().catch(() => { }); playbackContext = null; }
  nextPlaybackTime = 0;

  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
  }

  disableTransportSelect(false);

  // Clear video sources
  const videoEl = document.getElementById('video');
  if (videoEl) videoEl.srcObject = null;
  const canvasEl = document.getElementById('remote-canvas');
  if (canvasEl) {
    const ctx = canvasEl.getContext('2d');
    if (ctx) ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
  }
  const mediaDiv = document.getElementById('media');
  if (mediaDiv) mediaDiv.style.display = 'none';

  // Hide video overlay
  const overlay = document.getElementById('video-overlay');
  if (overlay) {
    overlay.classList.add('hidden');
    overlay.textContent = '';
  }

  const container = document.getElementById('detection-boxes-container');
  if (container) {
    container.innerHTML = '';
  }
  const labelsContainer = document.getElementById('detection-labels-container');
  if (labelsContainer) {
    labelsContainer.innerHTML = '';
  }
}

enumerateInputDevices();

// Keyboard control
var controlEnabled = false;

function toggleControl() {
  controlEnabled = !controlEnabled;
  const btn = document.getElementById('enable-control');
  if (btn) {
    if (controlEnabled) {
      btn.className = 'w-full bg-green-600 text-white px-4 py-2 rounded-lg font-medium hover:bg-green-700 transition-colors flex items-center justify-center space-x-2';
      const span = btn.querySelector('span');
      if (span) span.textContent = 'Control Enabled';
    } else {
      btn.className = 'w-full bg-gray-200 text-gray-700 px-4 py-2 rounded-lg font-medium hover:bg-gray-300 transition-colors flex items-center justify-center space-x-2';
      const span = btn.querySelector('span');
      if (span) span.textContent = 'Enable Control';
    }
  }
}

const ARROW_KEYS = new Set(['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight']);

function sendKeyEvent(type, evt) {
  if (!controlEnabled) return;
  // Prevent arrow keys from scrolling the page while control is active
  if (ARROW_KEYS.has(evt.key)) evt.preventDefault();
  sendControlMessage({ type, key: evt.key, code: evt.code });
}

document.addEventListener('keydown', (evt) => sendKeyEvent('keydown', evt));
document.addEventListener('keyup', (evt) => sendKeyEvent('keyup', evt));

// Automatically enable video checkboxes when a video model provider is selected
const handleProviderChange = () => {
  const simliChecked = document.getElementById('provider-simli')?.checked;
  const yoloChecked = document.getElementById('provider-yolo')?.checked;
  const faceLandmarkerChecked = document.getElementById('provider-face-landmarker')?.checked;
  const sam2Checked = document.getElementById('provider-sam2')?.checked;
  const sam3Checked = document.getElementById('provider-sam3')?.checked;
  const inceptionChecked = document.getElementById('provider-inception')?.checked;
  const externalWebsocketChecked = document.getElementById('provider-external-websocket')?.checked;
  const mujocoChecked = document.getElementById('provider-mujoco')?.checked;
  const geminiRoboticsChecked = document.getElementById('provider-gemini-robotics')?.checked;
  const cosmosChecked = document.getElementById('provider-cosmos')?.checked;

  if (simliChecked || yoloChecked || faceLandmarkerChecked || sam2Checked || sam3Checked || inceptionChecked || geminiRoboticsChecked || cosmosChecked) {
    const sendVideo = document.getElementById('send-video');
    const recvVideo = document.getElementById('recv-video');
    if (sendVideo) sendVideo.checked = true;
    if (recvVideo) recvVideo.checked = true;
  }

  if (externalWebsocketChecked || mujocoChecked) {
    const recvVideo = document.getElementById('recv-video');
    if (recvVideo) recvVideo.checked = true;
  }
};

document.getElementById('provider-simli')?.addEventListener('change', handleProviderChange);
document.getElementById('provider-yolo')?.addEventListener('change', handleProviderChange);
document.getElementById('provider-face-landmarker')?.addEventListener('change', handleProviderChange);
document.getElementById('provider-sam2')?.addEventListener('change', handleProviderChange);
document.getElementById('provider-sam3')?.addEventListener('change', handleProviderChange);
document.getElementById('provider-inception')?.addEventListener('change', handleProviderChange);
document.getElementById('provider-gemini-robotics')?.addEventListener('change', handleProviderChange);
document.getElementById('provider-external-websocket')?.addEventListener('change', handleProviderChange);
document.getElementById('provider-mujoco')?.addEventListener('change', handleProviderChange);
document.getElementById('provider-cosmos')?.addEventListener('change', handleProviderChange);

// --- Active sessions ---
var currentSessionId = null;

function renderSessions(sessions) {
  const list = document.getElementById('sessions-list');
  if (!list) return;
  list.innerHTML = '';

  if (!sessions.length) {
    const empty = document.createElement('p');
    empty.className = 'text-sm text-gray-400';
    empty.textContent = 'No active sessions.';
    list.appendChild(empty);
    return;
  }

  sessions.forEach((session) => {
    const row = document.createElement('div');
    row.className = 'flex items-center justify-between border rounded-lg px-3 py-2 bg-gray-50';

    const info = document.createElement('a');
    info.href = `session.html?sessionId=${session.id}`;
    info.target = '_blank';
    info.className = 'hover:underline cursor-pointer block';

    const title = document.createElement('div');
    title.className = 'text-sm font-medium text-primary font-mono';
    title.textContent = session.id.slice(0, 8) + (session.id === currentSessionId ? ' (you)' : '');
    const modelCount = session.connections.reduce((sum, c) => sum + (c.models ? c.models.length : 0), 0);
    const subtitle = document.createElement('div');
    subtitle.className = 'text-xs text-gray-500';
    subtitle.textContent = `${session.connections.length} connection${session.connections.length === 1 ? '' : 's'}, ${modelCount} model${modelCount === 1 ? '' : 's'}`;
    info.appendChild(title);
    info.appendChild(subtitle);
    row.appendChild(info);

    if (session.id !== currentSessionId) {
      const btn = document.createElement('button');
      btn.className = 'bg-primary text-white text-sm px-3 py-1.5 rounded-lg hover:bg-blue-700 transition-colors';
      btn.textContent = 'Join';
      btn.onclick = () => joinSession(session.id).catch(alert);
      row.appendChild(btn);
    }

    list.appendChild(row);
  });
}

function renderSessionsError() {
  const list = document.getElementById('sessions-list');
  if (!list) return;
  list.innerHTML = '';

  const error = document.createElement('p');
  error.className = 'text-sm text-red-500';
  error.textContent = 'Server Unavailable';
  list.appendChild(error);
}

async function refreshSessions() {
  try {
    const response = await fetch(`${BASE_URL}/session`);
    if (!response.ok) {
      renderSessionsError();
      return;
    }
    const data = await response.json();
    renderSessions(data.sessions || []);
  } catch (e) {
    renderSessionsError();
  }
}

// --- Session connections/models detail (session.html) ---

function renderSessionConnections(containerId, connections) {
  const container = document.getElementById(containerId);
  if (!container) return;
  container.innerHTML = '';

  if (!connections.length) {
    const empty = document.createElement('p');
    empty.className = 'text-sm text-gray-400';
    empty.textContent = 'No connections.';
    container.appendChild(empty);
    return;
  }

  connections.forEach((connection) => {
    const card = document.createElement('div');
    card.className = 'border rounded-lg p-4 bg-gray-50 space-y-2';

    const title = document.createElement('div');
    title.className = 'text-sm font-medium text-primary font-mono';
    title.textContent = connection.id;
    card.appendChild(title);

    if (!connection.models.length) {
      const noModels = document.createElement('p');
      noModels.className = 'text-xs text-gray-400 pl-2';
      noModels.textContent = 'No models loaded.';
      card.appendChild(noModels);
    } else {
      connection.models.forEach((model) => {
        const modelBlock = document.createElement('div');
        modelBlock.className = 'pl-4 border-l-2 border-primary/30 space-y-1';

        const modelName = document.createElement('div');
        modelName.className = 'text-sm font-semibold text-gray-800';
        modelName.textContent = model.name;
        modelBlock.appendChild(modelName);

        const propKeys = Object.keys(model.properties || {});
        if (propKeys.length) {
          const propsList = document.createElement('dl');
          propsList.className = 'text-xs text-gray-600 space-y-0.5';
          propKeys.forEach((key) => {
            const row = document.createElement('div');
            row.className = 'flex space-x-2';
            const dt = document.createElement('dt');
            dt.className = 'font-medium text-gray-500 shrink-0';
            dt.textContent = `${key}:`;
            const dd = document.createElement('dd');
            dd.className = 'break-all';
            dd.textContent = String(model.properties[key]);
            row.appendChild(dt);
            row.appendChild(dd);
            propsList.appendChild(row);
          });
          modelBlock.appendChild(propsList);
        }

        card.appendChild(modelBlock);
      });
    }

    container.appendChild(card);
  });
}

async function refreshSessionConnections(containerId, sessionId) {
  try {
    const response = await fetch(`${BASE_URL}/session`);
    if (!response.ok) return;
    const data = await response.json();
    const session = (data.sessions || []).find((s) => s.id === sessionId);
    renderSessionConnections(containerId, session ? session.connections : []);
  } catch (e) {
    // Ignore transient errors and keep the last rendered state.
  }
}

// Join an existing session as a viewer/participant: receive its audio, video
// and data channel events (and send mic audio when audio is enabled).
// Joining an existing session is only supported over WebRTC.
async function joinSession(sessionId) {
  if (pc) {
    alert('Already connected. Stop the current session first.');
    return;
  }

  const startBtn = document.getElementById('start');
  if (startBtn) startBtn.style.display = 'none';

  resetConversationLog();

  activeTransport = 'webrtc';
  setVideoDisplayMode('webrtc');
  disableTransportSelect(true);

  // Make sure incoming video tracks get routed to the video element
  const recvVideo = document.getElementById('recv-video');
  if (recvVideo) recvVideo.checked = true;

  pc = createPeerConnection();

  dc = pc.createDataChannel('data', { ordered: true });
  dc.addEventListener('message', handleDataChannelMessage);

  const useAudio = document.getElementById('use-audio');
  if (useAudio && useAudio.checked) {
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaStream.getTracks().forEach((track) => pc.addTrack(track, mediaStream));
  } else {
    pc.addTransceiver('audio', { direction: 'recvonly' });
  }

  const vt = pc.addTransceiver('video', { direction: 'recvonly' });
  // Prefer H264; strip VP8 so it isn't offered at all.
  const caps = RTCRtpReceiver.getCapabilities?.('video');
  if (caps) {
    const h264 = caps.codecs.filter(c => c.mimeType === 'video/H264');
    const rest = caps.codecs.filter(c => c.mimeType !== 'video/H264' && c.mimeType !== 'video/VP8');
    const rtx = caps.codecs.filter(c => c.mimeType === 'video/rtx');
    try { vt.setCodecPreferences([...h264, ...rest, ...rtx]); } catch (_) { }
  }
  const mediaDiv = document.getElementById('media');
  if (mediaDiv) mediaDiv.style.display = 'block';

  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);

  let result;
  try {
    const response = await fetch(`${BASE_URL}/session/${sessionId}/connection`, {
      body: JSON.stringify({ sdp: offer.sdp }),
      headers: { 'Content-Type': 'application/json' },
      method: 'POST'
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    result = await response.json();
  } catch (error) {
    console.error('Join session error:', error);
    alert(`Failed to join session: ${error.message}`);
    await stop();
    return;
  }

  currentSessionId = result.sessionId;
  await pc.setRemoteDescription(new RTCSessionDescription({
    type: 'answer',
    sdp: result.sdp,
  }));

  const stopBtn = document.getElementById('stop');
  if (stopBtn) stopBtn.style.display = 'inline-block';
  refreshSessions();
}

// Poll the active sessions every 10 seconds
setInterval(refreshSessions, 10000);
refreshSessions();

// On session.html, poll the connections/models detail for the viewed session.
if (document.getElementById('session-connections-list')) {
  const viewedSessionId = new URLSearchParams(window.location.search).get('sessionId');
  if (viewedSessionId) {
    refreshSessionConnections('session-connections-list', viewedSessionId);
    setInterval(() => refreshSessionConnections('session-connections-list', viewedSessionId), 5000);
  } else {
    document.getElementById('session-connections-list').innerHTML =
      '<p class="text-sm text-gray-400">No session ID provided.</p>';
  }
}
