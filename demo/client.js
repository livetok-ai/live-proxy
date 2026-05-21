// get DOM elements
const dataChannelLog = document.getElementById('data-channel'),
  iceConnectionLog = document.getElementById('ice-connection-state');

var pc = null;
var dc = null;
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
  let model = document.getElementById('model').value;
  if (document.getElementById('provider-simli')?.checked) {
    model += ';simli';
  }
  if (document.getElementById('provider-yolo')?.checked) {
    model += ';yolo-overlay';
  }
  if (document.getElementById('provider-face-landmarker')?.checked) {
    model += ';face_landmarker-overlay';
  }
  if (document.getElementById('provider-text-sentiment')?.checked) {
    model += ';text_sentiment';
  }
  const systemInstructions = document.getElementById('system-instructions').value.trim();
  const tools = JSON.parse(document.getElementById('tools').value.trim());
  const voice = document.getElementById('voice').value;
  const language = document.getElementById('language').value;
  const apiKey = document.getElementById('api-key').value.trim();
  const avatar = document.getElementById('recv-video').checked;
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

  // Add avatar if provided
  if (avatar) {
    body.avatar = avatar;
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
  dc.addEventListener('close', () => {
    // Data channel closed - no need to show this to user
  });
  dc.addEventListener('open', () => {
    // Data channel opened - no need to show this to user
  });
  dc.addEventListener('message', (evt) => {
    const message = evt.data;

    try {
      // Try to parse as JSON
      const data = JSON.parse(message);

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
    pc.addTransceiver('video', { direction: 'recvonly' });
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

  // Clear video sources
  document.getElementById('video').srcObject = null;
  document.getElementById('media').style.display = 'none';
}

enumerateInputDevices();

// Automatically enable video checkboxes when a video model provider is selected
const handleProviderChange = () => {
  const simliChecked = document.getElementById('provider-simli')?.checked;
  const yoloChecked = document.getElementById('provider-yolo')?.checked;
  const faceLandmarkerChecked = document.getElementById('provider-face-landmarker')?.checked;

  if (simliChecked || yoloChecked || faceLandmarkerChecked) {
    const sendVideo = document.getElementById('send-video');
    const recvVideo = document.getElementById('recv-video');
    if (sendVideo) sendVideo.checked = true;
    if (recvVideo) recvVideo.checked = true;
  }
};

document.getElementById('provider-simli')?.addEventListener('change', handleProviderChange);
document.getElementById('provider-yolo')?.addEventListener('change', handleProviderChange);
document.getElementById('provider-face-landmarker')?.addEventListener('change', handleProviderChange);


