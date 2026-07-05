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

function createPeerConnection() {
  var config = {
  };

  pc = new RTCPeerConnection(config);

  pc.addEventListener('iceconnectionstatechange', () => {
    iceConnectionLog.textContent += ' -> ' + pc.iceConnectionState;
  }, false);
  iceConnectionLog.textContent = pc.iceConnectionState;

  // connect audio / video
  pc.addEventListener('track', (evt) => {
    if (evt.track.kind == 'video' && document.getElementById('recv-video').checked) {
      // Reduce jitter buffer to 100 ms for lower latency video playback
      if (evt.receiver && 'jitterBufferTarget' in evt.receiver) {
        evt.receiver.jitterBufferTarget = 0.1;
      }
      document.getElementById('video').srcObject = evt.streams[0];
    } else {
      document.getElementById('audio').srcObject = evt.streams[0];
    }
  });

  return pc;
}

function enumerateInputDevices() {
  const populateSelect = (select, devices) => {
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

  if (document.getElementById('provider-simli')?.checked) {
    appendAddon('simli');
  }
  if (document.getElementById('provider-yolo')?.checked) {
    appendAddon('yolo[draw=true,sampling=15]');
  }
  if (document.getElementById('provider-face-landmarker')?.checked) {
    appendAddon('face_landmarker[draw=true]');
  }
  if (document.getElementById('provider-text-sentiment')?.checked) {
    appendAddon('text_sentiment');
  }
  if (document.getElementById('provider-sam2')?.checked) {
    appendAddon('sam[version=sam2.1_t.pt,draw=true,sampling=15]');
  }
  if (document.getElementById('provider-sam3')?.checked) {
    appendAddon('sam[version=sam2.1_t.pt,draw=true,sampling=15]');
  }
  if (document.getElementById('provider-inception')?.checked) {
    appendAddon('inception');
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
  const answer = new RTCSessionDescription({
    type: 'answer',
    sdp: result.sdp,
  });
  await pc.setRemoteDescription(answer);
}

async function start() {
  document.getElementById('start').style.display = 'none';

  // Reset transcription tracking and clear log
  lastMessageElement = null;
  isAppendingToLast = false;
  dataChannelLog.innerHTML = '';

  pc = createPeerConnection();

  dc = pc.createDataChannel('data', { ordered: true });
  dc.addEventListener('close', () => {});
  dc.addEventListener('open', () => {});

  // Unreliable channel for low-latency key events (fire-and-forget)
  dcUnreliable = pc.createDataChannel('keys', { ordered: false, maxRetransmits: 0 });
  dcUnreliable.addEventListener('open', () => {
    heartbeatInterval = setInterval(() => {
      if (dcUnreliable && dcUnreliable.readyState === 'open') {
        try { dcUnreliable.send('{}'); } catch (_) {}
      }
    }, 20);
  });
  dc.addEventListener('message', (evt) => {
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
  });

  // Build media constraints.
  const useAudio = document.getElementById('use-audio').checked;
  const sendVideo = document.getElementById('send-video').checked;
  const recvVideo = document.getElementById('recv-video').checked;

  const constraints = {
    audio: false,
    video: false,
  };

  if (useAudio) {
    const audioConstraints = {};

    const device = document.getElementById('audio-input').value;
    if (device) {
      audioConstraints.deviceId = { exact: device };
    }

    constraints.audio = Object.keys(audioConstraints).length ? audioConstraints : true;
  }

  if (sendVideo) {
    const videoConstraints = { width: { max: 320 }, height: { max: 240 } };

    const device = document.getElementById('video-input').value;
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
      const rtx  = caps.codecs.filter(c => c.mimeType === 'video/rtx');
      try { vt.setCodecPreferences([...h264, ...rest, ...rtx]); } catch (_) {}
    }
    document.getElementById('media').style.display = 'block';
  }

  if (constraints.audio || constraints.video) {
    const stream = await navigator.mediaDevices.getUserMedia(constraints)
    stream.getTracks().forEach((track) => {
      if (track.kind === 'video') {
        track = track.clone();
        track.applyConstraints({
          frameRate: 5,
        });
      }
      pc.addTrack(track, stream);
    });
    if (constraints.video) {
      document.getElementById('media').style.display = 'block';
      document.getElementById('video').srcObject = stream;
    }
    await negotiate();
  } else {
    await negotiate();
  }

  document.getElementById('stop').style.display = 'inline-block';
}

async function stop() {
  document.getElementById('stop').style.display = 'none';
  document.getElementById('start').style.display = 'inline-block';

  if (pc) {
    pc.close();
    pc = null;
  }
  dc = null;
  dcUnreliable = null;
  if (heartbeatInterval) { clearInterval(heartbeatInterval); heartbeatInterval = null; }

  // Clear video sources
  document.getElementById('video').srcObject = null;
  document.getElementById('media').style.display = 'none';

  // Hide video overlay
  const overlay = document.getElementById('video-overlay');
  if (overlay) {
    overlay.classList.add('hidden');
    overlay.textContent = '';
  }
}

enumerateInputDevices();

// Keyboard control
var controlEnabled = false;

function toggleControl() {
  controlEnabled = !controlEnabled;
  const btn = document.getElementById('enable-control');
  if (controlEnabled) {
    btn.className = 'w-full bg-green-600 text-white px-4 py-2 rounded-lg font-medium hover:bg-green-700 transition-colors flex items-center justify-center space-x-2';
    btn.querySelector('span').textContent = 'Control Enabled';
  } else {
    btn.className = 'w-full bg-gray-200 text-gray-700 px-4 py-2 rounded-lg font-medium hover:bg-gray-300 transition-colors flex items-center justify-center space-x-2';
    btn.querySelector('span').textContent = 'Enable Control';
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

  if (simliChecked || yoloChecked || faceLandmarkerChecked || sam2Checked || sam3Checked || inceptionChecked) {
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
document.getElementById('provider-insivision')?.addEventListener('change', handleProviderChange);
document.getElementById('provider-mujoco')?.addEventListener('change', handleProviderChange);


