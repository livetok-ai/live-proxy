// get DOM elements
const dataChannelLog = document.getElementById('data-channel'),
    iceConnectionLog = document.getElementById('ice-connection-state');

var pc = null;
var dc = null;

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
        if (evt.track.kind == 'video')
            document.getElementById('video').srcObject = evt.streams[0];
        else
            document.getElementById('audio').srcObject = evt.streams[0];
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
    const model = document.getElementById('model').value;
    const systemInstructions = document.getElementById('system-instructions').value.trim();
    const callbackUrl = document.getElementById('callback-url').value.trim();
    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);

    let requestBody;
    let contentType;

    // Send JSON body with SDP and optional parameters
    const body = {
        sdp: offer.sdp
    };
    if (systemInstructions) {
        body.system_instructions = systemInstructions;
    }
    if (callbackUrl) {
        body.callback = callbackUrl;
    }
    requestBody = JSON.stringify(body);
    contentType = 'application/json';

    const response = await fetch(`/?model=${model}`, {
        body: requestBody,
        headers: {
            'Content-Type': contentType
        },
        method: 'POST'
    });

    const result = await response.json();
    const answer = new RTCSessionDescription({
        type: 'answer',
        sdp: result.sdp,
    });
    await pc.setRemoteDescription(answer);
}

async function start() {
    document.getElementById('start').style.display = 'none';

    pc = createPeerConnection();

    dc = pc.createDataChannel('data', { ordered: true });
    dc.addEventListener('close', () => {
        dataChannelLog.textContent += '- close\n';
    });
    dc.addEventListener('open', () => {
        dataChannelLog.textContent += '- open\n';
    });
    dc.addEventListener('message', (evt) => {
        dataChannelLog.textContent += '< ' + evt.data + '\n';
    });

    // Build media constraints.

    const constraints = {
        audio: false,
        video: false,
    };

    if (document.getElementById('use-audio').checked) {
        const audioConstraints = {};

        const device = document.getElementById('audio-input').value;
        if (device) {
            audioConstraints.deviceId = { exact: device };
        }

        constraints.audio = Object.keys(audioConstraints).length ? audioConstraints : true;
    }

    if (document.getElementById('use-video').checked) {
        const videoConstraints = { width: { max: 320 }, height: { max: 240 } };

        const device = document.getElementById('video-input').value;
        if (device) {
            videoConstraints.deviceId = { exact: device };
        }

        constraints.video = Object.keys(videoConstraints).length ? videoConstraints : true;
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

    if (pc) {
        pc.close();
        pc = null;
    }
}

enumerateInputDevices();

