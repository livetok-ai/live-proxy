import pytest

from connection import parse_model, split_models


def test_parse_model():
    # Simple model
    assert parse_model("yolo") == ("yolo", {})

    # Model with arguments
    assert parse_model("sam[version=sam2.1_t.pt,sampling=5]") == ("sam", {"version": "sam2.1_t.pt", "sampling": 5})

    # Model with quoted arguments
    assert parse_model("sam[version=\"sam2.1_t.pt\",device='cpu']") == (
        "sam",
        {"version": "sam2.1_t.pt", "device": "cpu"},
    )

    # Model with numbers
    assert parse_model("yolo[sampling=5]") == ("yolo", {"sampling": 5})
    assert parse_model("yolo[sampling=2.5]") == ("yolo", {"sampling": 2.5})


def test_split_models():
    # No brackets, split by comma/semicolon
    assert split_models("yolo,sam") == ["yolo", "sam"]
    assert split_models("yolo;sam") == ["yolo", "sam"]

    # With brackets, commas inside brackets are ignored
    assert split_models("sam[version=sam2.1_t.pt,sampling=5],yolo[sampling=5]") == [
        "sam[version=sam2.1_t.pt,sampling=5]",
        "yolo[sampling=5]",
    ]

    # Semi-colons and commas outside
    assert split_models("sam[version=sam2.1_t.pt,sampling=5];yolo") == ["sam[version=sam2.1_t.pt,sampling=5]", "yolo"]


@pytest.mark.asyncio
async def test_add_model_kwargs():
    from connection import Connection
    from providers.yolo import YoloProvider

    conn = Connection()
    with pytest.MonkeyPatch().context() as mp:

        async def mock_connect(self):
            pass

        mp.setattr(YoloProvider, "connect", mock_connect)

        m = await conn.add_model("yolo", {"sampling": 150})
        assert m is not None
        assert isinstance(m, YoloProvider)
        assert m.sampling_rate == 150


@pytest.mark.asyncio
async def test_connection_run_setup_before_connect():
    from connection import Connection
    from providers.inception import InceptionProvider
    from providers.yolo import YoloProvider

    # Track sequence of actions
    actions = []

    # Mock Model/Provider methods to track order
    class MockYoloProvider(YoloProvider):
        async def connect(self):
            actions.append(("connect", self.name))

        def on(self, event, handler):
            actions.append(("on", self.name, event))
            super().on(event, handler)

    class MockInceptionProvider(InceptionProvider):
        async def connect(self):
            actions.append(("connect", self.name))

        def on(self, event, handler):
            actions.append(("on", self.name, event))
            super().on(event, handler)

    # Override MODEL_MAP to use our mock
    from connection import MODEL_MAP

    original_yolo = MODEL_MAP["yolo"]
    original_inception = MODEL_MAP["inception"]
    MODEL_MAP["yolo"] = MockYoloProvider
    MODEL_MAP["inception"] = MockInceptionProvider

    async def mock_run_setup(connection):
        actions.append(("setup_scripts",))
        # Add a dynamic model during setup
        connection.add_model_sync("inception")

    import script_manager

    original_run_setup = script_manager.run_setup

    try:
        script_manager.run_setup = mock_run_setup

        class MockPC:
            def __init__(self):
                self.localDescription = type("LocalDesc", (), {"sdp": "v=0\r\n"})()

            def on(self, event):
                def decorator(func):
                    return func

                return decorator

            def getTransceivers(self):
                return []

            def addTrack(self, track):
                pass

            async def setRemoteDescription(self, desc):
                pass

            async def createAnswer(self):
                return type("Answer", (), {})()

            async def setLocalDescription(self, desc):
                pass

        conn = Connection()
        # Mock pc so start() can run WebRTC setup
        conn.pc = MockPC()

        sdp = "v=0\r\no=- 0 0 IN IP4 127.0.0.1\r\ns=-\r\nt=0 0\r\nc=IN IP4 127.0.0.1\r\nm=audio 9 RTP/AVP 0\r\n"
        await conn.start(sdp=sdp, model="yolo")

        # Verify setup runs before connect
        setup_index = [i for i, a in enumerate(actions) if a[0] == "setup_scripts"][0]
        connect_indices = [i for i, a in enumerate(actions) if a[0] == "connect"]

        # All connects must happen AFTER setup
        for c_idx in connect_indices:
            assert c_idx > setup_index

        # Verify default yolo and dynamic inception both got connected
        connected_models = [a[1] for a in actions if a[0] == "connect"]
        assert "yolo" in connected_models
        assert "inception" in connected_models

    finally:
        MODEL_MAP["yolo"] = original_yolo
        MODEL_MAP["inception"] = original_inception
        script_manager.run_setup = original_run_setup


def test_broadcast_sip_rtmp():
    from connection import Connection

    # 1. SIP connection
    class SimplePeerConnection:
        pass

    conn_sip = Connection()
    conn_sip.pc = SimplePeerConnection()

    trace_logs = []
    warn_logs = []
    conn_sip.trace = lambda msg, *args: trace_logs.append(msg % args if args else msg)
    conn_sip.warn = lambda msg, *args: warn_logs.append(msg % args if args else msg)

    conn_sip._broadcast_to_channels("test-event-sip", publish=False)
    assert len(warn_logs) == 0
    assert len(trace_logs) == 1
    assert "Event: test-event-sip" in trace_logs[0]

    # 2. RTMP connection
    class RTMPPeerConnection:
        pass

    conn_rtmp = Connection()
    conn_rtmp.pc = RTMPPeerConnection()

    trace_logs_rtmp = []
    warn_logs_rtmp = []
    conn_rtmp.trace = lambda msg, *args: trace_logs_rtmp.append(msg % args if args else msg)
    conn_rtmp.warn = lambda msg, *args: warn_logs_rtmp.append(msg % args if args else msg)

    conn_rtmp._broadcast_to_channels("test-event-rtmp", publish=False)
    assert len(warn_logs_rtmp) == 0
    assert len(trace_logs_rtmp) == 1
    assert "Event: test-event-rtmp" in trace_logs_rtmp[0]

    # 3. Regular Connection (WebRTC / WebTransport)
    class OtherConnection:
        pass

    conn_other = Connection()
    conn_other.pc = OtherConnection()

    trace_logs_other = []
    warn_logs_other = []
    conn_other.trace = lambda msg, *args: trace_logs_other.append(msg % args if args else msg)
    conn_other.warn = lambda msg, *args: warn_logs_other.append(msg % args if args else msg)

    conn_other._broadcast_to_channels("test-event-other", publish=False)
    assert len(warn_logs_other) == 1
    assert "Could not broadcast: no open data channels" in warn_logs_other[0]
    assert len(trace_logs_other) == 0
