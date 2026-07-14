// get DOM elements
const dataChannelLog = document.getElementById('data-channel'),
  iceConnectionLog = document.getElementById('ice-connection-state');

var pc = null;
var dc = null;        // reliable datachannel (transcriptions, control messages)
var dcUnreliable = null; // unreliable datachannel (key events)
var heartbeatInterval = null;
var lastMessageElement = null; // Track the last message bubble element
var isAppendingToLast = false; // Track if we're appending to the last message

// Read from query params
const urlParams = new URLSearchParams(window.location.search);
const BASE_URL = urlParams.get('base_url') || '';

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

function renderObjects(objects) {
  const container = document.getElementById('detection-boxes-container');
  if (!container) return;

  // Clear previous boxes
  container.innerHTML = '';

  if (!objects || !Array.isArray(objects)) return;

  objects.forEach(obj => {
    const { label, top, left, bottom, right } = obj;
    if (top === undefined || left === undefined || bottom === undefined || right === undefined) return;

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

function handleDataChannelMessage(evt) {
  const message = evt.data;

  try {
    // Try to parse as JSON
    const data = JSON.parse(message);

    // Show on top of the video feed any data received that has a "display" attribute
    if (data && typeof data === 'object' && 'display' in data) {
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
      const currentMessageType = data.role; // 'user' or 'model'
      const content = data.content;

      addOrAppendMessage(currentMessageType, content);
    } else if (data.type === 'inference') {
      // Latest raw inference result for a processed frame — replaces
      // whatever was previously shown, it is not appended.
      updateInferenceStatus(data.provider, data.content);
    } else if (data.type === 'objects') {
      renderObjects(data.objects);
    } else {
      // Handle other JSON message types if needed - ignore for now
    }
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
      if (videoEl) videoEl.srcObject = evt.streams[0];
    } else {
      const audioEl = document.getElementById('audio');
      if (audioEl) audioEl.srcObject = evt.streams[0];
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

async function negotiate() {
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
    appendAddon(`yolo[draw=false,sampling=${sampling}]`);
  }
  if (document.getElementById('provider-face-landmarker')?.checked) {
    appendAddon('face_landmarker[draw=false]');
  }
  if (document.getElementById('provider-text-sentiment')?.checked) {
    appendAddon('text_sentiment');
  }
  if (document.getElementById('provider-sam2')?.checked) {
    appendAddon(`sam[version=sam2.1_t.pt,draw=false,sampling=${sampling}]`);
  }
  if (document.getElementById('provider-sam3')?.checked) {
    appendAddon(`sam[version=sam2.1_t.pt,draw=false,sampling=${sampling}]`);
  }
  if (document.getElementById('provider-inception')?.checked) {
    appendAddon('inception');
  }
  if (document.getElementById('provider-gemini-robotics')?.checked) {
    appendAddon(`gemini-robotics[draw=false,sampling=${sampling}]`);
  }
  if (document.getElementById('provider-insivision')?.checked) {
    appendAddon('insivision');
  }
  if (document.getElementById('provider-mujoco')?.checked) {
    appendAddon('mujoco');
  }
  const systemInstructions = document.getElementById('system-instructions').value.trim();
  const tools = JSON.parse(document.getElementById('tools').value.trim());
  const voice = document.getElementById('voice').value;
  const language = document.getElementById('language').value;
  const apiKey = document.getElementById('api-key').value.trim();
  const ragCorpus = document.getElementById('rag-corpus').value.trim();
  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);

  let requestBody;
  let contentType;

  // Send JSON body with SDP and optional parameters
  const body = {
    sdp: offer.sdp,
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
  };

  // Add API key if provided
  if (apiKey) {
    body.api_key = apiKey;
  }

  // Add RAG corpus if provided
  if (ragCorpus) {
    body.rag_corpus = ragCorpus;
  }

  requestBody = JSON.stringify(body);
  contentType = 'application/json';

  let response;
  let result;

  try {
    response = await fetch(`${BASE_URL}/connection`, {
      body: requestBody,
      headers: {
        'Content-Type': contentType
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

async function start() {
  const startBtn = document.getElementById('start');
  if (startBtn) startBtn.style.display = 'none';

  // Reset transcription tracking and clear log
  lastMessageElement = null;
  isAppendingToLast = false;
  if (dataChannelLog) dataChannelLog.innerHTML = '';

  pc = createPeerConnection();

  dc = pc.createDataChannel('data', { ordered: true });
  dc.addEventListener('close', () => { });
  dc.addEventListener('open', () => { });

  // Unreliable channel for low-latency key events (fire-and-forget)
  dcUnreliable = pc.createDataChannel('keys', { ordered: false, maxRetransmits: 0 });
  dcUnreliable.addEventListener('open', () => {
    heartbeatInterval = setInterval(() => {
      if (dcUnreliable && dcUnreliable.readyState === 'open') {
        try { dcUnreliable.send('{}'); } catch (_) { }
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
    const stream = await navigator.mediaDevices.getUserMedia(constraints)
    stream.getTracks().forEach((track) => {
      pc.addTrack(track, stream);
    });
    if (constraints.video) {
      const mediaDiv = document.getElementById('media');
      if (mediaDiv) mediaDiv.style.display = 'block';
      const videoEl = document.getElementById('video');
      if (videoEl) videoEl.srcObject = stream;
    }
    await negotiate();
  } else {
    await negotiate();
  }

  const stopBtn = document.getElementById('stop');
  if (stopBtn) stopBtn.style.display = 'inline-block';
}

async function stop() {
  const stopBtn = document.getElementById('stop');
  if (stopBtn) stopBtn.style.display = 'none';
  const startBtn = document.getElementById('start');
  if (startBtn) startBtn.style.display = 'inline-block';

  currentSessionId = null;
  refreshSessions();

  if (pc) {
    pc.close();
    pc = null;
  }
  dc = null;
  dcUnreliable = null;
  if (heartbeatInterval) { clearInterval(heartbeatInterval); heartbeatInterval = null; }

  // Clear video sources
  const videoEl = document.getElementById('video');
  if (videoEl) videoEl.srcObject = null;
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
  const ch = dcUnreliable && dcUnreliable.readyState === 'open' ? dcUnreliable : dc;
  if (!ch || ch.readyState !== 'open') return;
  ch.send(JSON.stringify({ type, key: evt.key, code: evt.code }));
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
  const insivisionChecked = document.getElementById('provider-insivision')?.checked;
  const mujocoChecked = document.getElementById('provider-mujoco')?.checked;
  const geminiRoboticsChecked = document.getElementById('provider-gemini-robotics')?.checked;

  if (simliChecked || yoloChecked || faceLandmarkerChecked || sam2Checked || sam3Checked || inceptionChecked || geminiRoboticsChecked) {
    const sendVideo = document.getElementById('send-video');
    const recvVideo = document.getElementById('recv-video');
    if (sendVideo) sendVideo.checked = true;
    if (recvVideo) recvVideo.checked = true;
  }

  if (insivisionChecked || mujocoChecked) {
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
document.getElementById('provider-insivision')?.addEventListener('change', handleProviderChange);
document.getElementById('provider-mujoco')?.addEventListener('change', handleProviderChange);

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
    const subtitle = document.createElement('div');
    subtitle.className = 'text-xs text-gray-500';
    subtitle.textContent = `${session.connections.length} connection${session.connections.length === 1 ? '' : 's'}`;
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

// Join an existing session as a viewer/participant: receive its audio, video
// and data channel events (and send mic audio when audio is enabled).
async function joinSession(sessionId) {
  if (pc) {
    alert('Already connected. Stop the current session first.');
    return;
  }

  const startBtn = document.getElementById('start');
  if (startBtn) startBtn.style.display = 'none';

  // Reset transcription tracking and clear log
  lastMessageElement = null;
  isAppendingToLast = false;
  if (dataChannelLog) dataChannelLog.innerHTML = '';

  // Make sure incoming video tracks get routed to the video element
  const recvVideo = document.getElementById('recv-video');
  if (recvVideo) recvVideo.checked = true;

  pc = createPeerConnection();

  dc = pc.createDataChannel('data', { ordered: true });
  dc.addEventListener('message', handleDataChannelMessage);

  const useAudio = document.getElementById('use-audio');
  if (useAudio && useAudio.checked) {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    stream.getTracks().forEach((track) => pc.addTrack(track, stream));
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


