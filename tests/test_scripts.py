import os
from unittest.mock import patch

import pytest

import script_manager
from connection import Connection
from providers.face_landmarker import FaceLandmarkerProvider
from providers.inception import InceptionProvider
from providers.yolo import YoloProvider


class DummyInceptionModel(InceptionProvider):
    def __init__(self):
        self.input_enabled = True
        self.output_enabled = True
        self._handlers = {}

    def on(self, event, handler):
        self._handlers[event] = handler

    def off(self, event, handler):
        if event in self._handlers:
            del self._handlers[event]

    def trigger(self, event, data):
        if event in self._handlers:
            self._handlers[event](data)

    def enable_input(self):
        self.input_enabled = True

    def disable_input(self):
        self.input_enabled = False

    def enable_output(self):
        self.output_enabled = True

    def disable_output(self):
        self.output_enabled = False


class DummyYoloModel(YoloProvider):
    def __init__(self):
        self.input_enabled = True
        self.output_enabled = True
        self._handlers = {}

    def on(self, event, handler):
        self._handlers[event] = handler

    def off(self, event, handler):
        if event in self._handlers:
            del self._handlers[event]

    def trigger(self, event, data):
        if event in self._handlers:
            self._handlers[event](data)

    def enable_input(self):
        self.input_enabled = True

    def disable_input(self):
        self.input_enabled = False

    def enable_output(self):
        self.output_enabled = True

    def disable_output(self):
        self.output_enabled = False


class DummyFaceLandmarkerModel(FaceLandmarkerProvider):
    def __init__(self):
        self.input_enabled = True
        self.output_enabled = True
        self._handlers = {}

    def on(self, event, handler):
        self._handlers[event] = handler

    def off(self, event, handler):
        if event in self._handlers:
            del self._handlers[event]

    def trigger(self, event, data):
        if event in self._handlers:
            self._handlers[event](data)

    def enable_input(self):
        self.input_enabled = True

    def disable_input(self):
        self.input_enabled = False

    def enable_output(self):
        self.output_enabled = True

    def disable_output(self):
        self.output_enabled = False


@pytest.fixture(autouse=True)
def manage_counter_scripts():
    scripts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")

    src_py = os.path.join(scripts_dir, "counter._py")
    dst_py = os.path.join(scripts_dir, "counter.py")

    src_js = os.path.join(scripts_dir, "counter._js")
    dst_js = os.path.join(scripts_dir, "counter.js")

    copied_py = False
    copied_js = False

    if os.path.exists(src_py) and not os.path.exists(dst_py):
        import shutil

        shutil.copyfile(src_py, dst_py)
        copied_py = True

    if os.path.exists(src_js) and not os.path.exists(dst_js):
        import shutil

        shutil.copyfile(src_js, dst_js)
        copied_js = True

    yield

    if copied_py and os.path.exists(dst_py):
        try:
            os.remove(dst_py)
        except Exception:
            pass

    if copied_js and os.path.exists(dst_js):
        try:
            os.remove(dst_js)
        except Exception:
            pass


@pytest.mark.asyncio
async def test_get_model():
    """Test Connection.get_model returns the correct model from self.models."""
    conn = Connection()

    yolo_model = DummyYoloModel()
    face_model = DummyFaceLandmarkerModel()
    conn.models = [yolo_model, face_model]

    assert conn.get_model("yolo") is yolo_model
    assert conn.get_model("face_landmarker") is face_model
    assert conn.get_model("nonexistent") is None


@pytest.mark.asyncio
async def test_yolo_face_landmarker_enable_disable():
    """Test enable/disable methods and input/output flags on YOLO/FaceLandmarker providers."""
    yolo = YoloProvider()
    face = FaceLandmarkerProvider()

    assert yolo.input_enabled is True
    assert yolo.output_enabled is True
    assert face.input_enabled is True
    assert face.output_enabled is True

    yolo.disable_input()
    yolo.disable_output()
    face.disable_input()
    face.disable_output()
    assert yolo.input_enabled is False
    assert yolo.output_enabled is False
    assert face.input_enabled is False
    assert face.output_enabled is False

    yolo.enable_input()
    yolo.enable_output()
    face.enable_input()
    face.enable_output()
    assert yolo.input_enabled is True
    assert yolo.output_enabled is True
    assert face.input_enabled is True
    assert face.output_enabled is True


@pytest.mark.asyncio
async def test_script_loading():
    """Test load_all_scripts dynamically loads the script modules in scripts directory."""
    script_manager.load_all_scripts()
    assert len(script_manager.LOADED_SCRIPTS) > 0
    # Make sure counter script is loaded
    counter_loaded = any(m.__name__ == "counter" for m in script_manager.LOADED_SCRIPTS)
    assert counter_loaded is True
    # Make sure counter.js is loaded
    js_counter_loaded = any(m.__name__ == "counter.js" for m in script_manager.LOADED_SCRIPTS)
    assert js_counter_loaded is True


@pytest.mark.asyncio
async def test_counter_script_logic():
    """Test setup, objects detected (person vs other), sentiment detected, and teardown."""
    conn = Connection()

    yolo = DummyYoloModel()
    landmark = DummyFaceLandmarkerModel()

    conn.models = [yolo, landmark]

    # Load scripts and find counter module
    script_manager.load_all_scripts()
    counter_script = next(m for m in script_manager.LOADED_SCRIPTS if m.__name__ == "counter")

    # Run setup
    await counter_script.setup(conn)

    # Check that handlers were registered
    assert "objects_detected" in yolo._handlers
    assert "emotions_detected" in landmark._handlers

    # Check initial default state set by setup: YOLO active, Landmark inactive
    assert yolo.input_enabled is True
    assert yolo.output_enabled is True
    assert landmark.input_enabled is False
    assert landmark.output_enabled is False

    # 1. Trigger objects detected (no person)
    yolo.trigger("objects_detected", ["car", "dog"])
    assert yolo.output_enabled is True
    assert landmark.input_enabled is False
    assert landmark.output_enabled is False
    assert conn.object_counts == {"car": 1, "dog": 1}

    # 2. Trigger objects detected (with person)
    yolo.trigger("objects_detected", ["person", "chair"])
    assert yolo.output_enabled is False
    assert landmark.input_enabled is True
    assert landmark.output_enabled is True
    assert conn.object_counts == {"car": 1, "dog": 1, "person": 1, "chair": 1}

    # 3. Trigger sentiment detected
    landmark.trigger("emotions_detected", "happy")
    landmark.trigger("emotions_detected", "happy")
    landmark.trigger("emotions_detected", "neutral")
    assert conn.sentiment_counts == {"happy": 2, "neutral": 1}

    # 4. Run teardown
    with patch("logger.log_info") as mock_log_info:
        counter_script.teardown(conn)

        # Verify log output called for final counts
        log_messages = [call[0][0] for call in mock_log_info.call_args_list]
        assert any("Final object counts:" in msg for msg in log_messages)
        assert any("Final sentiment counts:" in msg for msg in log_messages)

    # Check event handlers were cleaned up
    assert "objects_detected" not in yolo._handlers
    assert "emotions_detected" not in landmark._handlers


@pytest.mark.asyncio
async def test_js_counter_script_logic():
    """Test JS setup, objects detected (person vs other), sentiment detected, and teardown."""
    conn = Connection()

    yolo = DummyYoloModel()
    landmark = DummyFaceLandmarkerModel()

    conn.models = [yolo, landmark]

    # Load scripts and find counter JS wrapper
    script_manager.load_all_scripts()
    counter_script = next(m for m in script_manager.LOADED_SCRIPTS if m.__name__ == "counter.js")

    # Run setup
    counter_script.setup(conn)
    # Set start_time to 0 to bypass the 5-second delay in test
    counter_script.contexts[conn.id][0].eval("connection.start_time = 0;")

    # Check that handlers were registered
    assert "objects_detected" in yolo._handlers
    assert "emotions_detected" in landmark._handlers

    # Check initial default state set by setup: YOLO active, Landmark inactive
    assert yolo.input_enabled is True
    assert yolo.output_enabled is True
    assert landmark.input_enabled is False
    assert landmark.output_enabled is False

    # 1. Trigger objects detected (no person)
    yolo.trigger("objects_detected", ["car", "dog"])
    assert yolo.output_enabled is True
    assert landmark.input_enabled is False
    assert landmark.output_enabled is False

    # 2. Trigger objects detected (with person)
    yolo.trigger("objects_detected", ["person", "chair"])
    assert yolo.output_enabled is False
    assert landmark.input_enabled is True
    assert landmark.output_enabled is True

    # 3. Trigger sentiment detected
    landmark.trigger("emotions_detected", "happy")
    landmark.trigger("emotions_detected", "happy")
    landmark.trigger("emotions_detected", "neutral")

    # No teardown check since it was removed from counter.js


@pytest.mark.asyncio
async def test_connection_send_data_python():
    """Test Connection.send_data is callable and correctly serializes and sends data via the data channel."""
    import json

    conn = Connection()

    class MockDataChannel:
        def __init__(self):
            self.readyState = "open"
            self.sent_messages = []

        def send(self, msg):
            self.sent_messages.append(msg)

    mock_dc = MockDataChannel()
    conn.data_channel = mock_dc

    # Send arbitrary dict
    test_dict = {"status": "ok", "count": 42}
    conn.send_data(test_dict)

    assert len(mock_dc.sent_messages) == 1
    assert json.loads(mock_dc.sent_messages[0]) == test_dict


@pytest.mark.asyncio
async def test_connection_send_data_js(tmp_path):
    """Test that connection.send_data() can be called from a JavaScript script and sends data."""
    import json

    conn = Connection()

    class MockDataChannel:
        def __init__(self):
            self.readyState = "open"
            self.sent_messages = []

        def send(self, msg):
            self.sent_messages.append(msg)

    mock_dc = MockDataChannel()
    conn.data_channel = mock_dc

    # Write a simple JS script that calls connection.send_data()
    js_file = tmp_path / "test_send.js"
    js_file.write_text("""
        function setup(connection) {
            connection.send_data({display: "Hello from JS", value: 123});
        }
    """)

    js_script = script_manager.JavaScriptScript(str(js_file))
    js_script.setup(conn)

    assert len(mock_dc.sent_messages) == 1
    sent_data = json.loads(mock_dc.sent_messages[0])
    assert sent_data == {"display": "Hello from JS", "value": 123}


@pytest.mark.asyncio
async def test_js_add_model_kwargs(tmp_path):
    """Test that connection.add_model() can be called from a JavaScript script with a dictionary of kwargs."""
    conn = Connection()

    with patch.object(conn, "add_model_sync") as mock_add_model_sync:
        js_file = tmp_path / "test_add_model_kwargs.js"
        js_file.write_text("""
            function setup(connection) {
                connection.add_model("yolo", { sampling: 150 });
            }
        """)

        js_script = script_manager.JavaScriptScript(str(js_file))
        js_script.setup(conn)

        mock_add_model_sync.assert_called_once_with("yolo", {"sampling": 150})


@pytest.mark.asyncio
async def test_js_add_tool_and_fetch(tmp_path):
    """Test that llm.addTool() and fetch() can be called from JavaScript script, and the tool callback executes."""
    from unittest.mock import AsyncMock, MagicMock

    from providers.gemini import GeminiProvider

    conn = Connection()

    # Initialize a real Gemini instance or mock
    gemini_model = GeminiProvider()
    gemini_model.connect = AsyncMock()
    gemini_model.close = AsyncMock()
    gemini_model.session = MagicMock()
    conn.models = [gemini_model]

    # Write a simple JS script that calls addTool
    js_file = tmp_path / "test_add_tool.js"
    js_file.write_text("""
        function setup(connection) {
            const llm = connection.get_model("gemini");
            llm.addTool({
                name: "get_weather",
                description: "Get weather",
                parameters: [
                    { name: "location", type: "string" }
                ],
                callback: (location) => {
                    const res = fetch("https://api.example.com/weather?q=" + location);
                    return res.ok ? "Weather in " + location + " is sunny" : "Error";
                }
            });
        }
    """)

    # Mock py_fetch in script_manager or test it directly
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = b'{"status": "ok"}'
        mock_urlopen.return_value.__enter__.return_value = mock_response

        js_script = script_manager.JavaScriptScript(str(js_file))
        js_script.setup(conn)

        # Verify tool is registered on Gemini
        assert "get_weather" in gemini_model.tools
        tool_dict, callback = gemini_model.tools["get_weather"]
        assert tool_dict["name"] == "get_weather"
        assert tool_dict["description"] == "Get weather"

        # Execute the Python callback (which invokes the JS callback)
        result = await callback({"location": "Chicago"})
        assert result == "Weather in Chicago is sunny"

        # Verify urlopen was called
        mock_urlopen.assert_called_once()
        called_args = mock_urlopen.call_args[0]
        assert called_args[0].full_url == "https://api.example.com/weather?q=Chicago"


@pytest.mark.asyncio
async def test_dynamic_script_execution():
    """Test that a script string attached to the connection is compiled and executed by script_manager."""
    conn = Connection()
    conn.script = """
        function setup(connection) {
            connection.send_data({ event: "setup_done" });
        }
        function teardown(connection) {
            connection.send_data({ event: "teardown_done" });
        }
    """

    class MockDataChannel:
        def __init__(self):
            self.readyState = "open"
            self.sent_messages = []

        def send(self, msg):
            self.sent_messages.append(msg)

    mock_dc = MockDataChannel()
    conn.data_channel = mock_dc

    # Run setup
    await script_manager.run_setup(conn)

    # Check setup executed
    assert hasattr(conn, "_compiled_script")
    assert len(mock_dc.sent_messages) == 1
    assert "setup_done" in mock_dc.sent_messages[0]

    # Run teardown
    await script_manager.run_teardown(conn)
    assert len(mock_dc.sent_messages) == 2
    assert "teardown_done" in mock_dc.sent_messages[1]


@pytest.mark.asyncio
async def test_js_async_add_tool(tmp_path):
    """Test that an async JavaScript tool callback can be registered and resolved properly via Promise queue."""
    from unittest.mock import AsyncMock, MagicMock

    from providers.gemini import GeminiProvider

    conn = Connection()
    gemini_model = GeminiProvider()
    gemini_model.connect = AsyncMock()
    gemini_model.close = AsyncMock()
    gemini_model.session = MagicMock()
    conn.models = [gemini_model]

    # Write a simple JS script that calls addTool with an async callback using a Promise
    js_file = tmp_path / "test_async_tool.js"
    js_file.write_text("""
        function setup(connection) {
            const llm = connection.get_model("gemini");
            llm.addTool({
                name: "async_multiply",
                description: "Async multiplication",
                parameters: [
                    { name: "x", type: "number" },
                    { name: "y", type: "number" }
                ],
                callback: async (x, y) => {
                    // Simulating an async operation using a resolved Promise
                    const factor = await Promise.resolve(2);
                    return x * y * factor;
                }
            });
        }
    """)

    js_script = script_manager.JavaScriptScript(str(js_file))
    js_script.setup(conn)

    # Verify tool is registered on Gemini
    assert "async_multiply" in gemini_model.tools
    tool_dict, callback = gemini_model.tools["async_multiply"]
    assert tool_dict["name"] == "async_multiply"

    # Execute the Python callback (which invokes the JS callback asynchronously)
    result = await callback({"x": 3, "y": 7})
    assert result == 42


@pytest.mark.asyncio
async def test_js_connection_close(tmp_path):
    """Test that connection.close() can be called from a JavaScript script and initiates connection closing."""
    import asyncio

    conn = Connection()

    # Write a simple JS script that calls connection.close()
    js_file = tmp_path / "test_close.js"
    js_file.write_text("""
        function setup(connection) {
            connection.close();
        }
    """)

    js_script = script_manager.JavaScriptScript(str(js_file))
    js_script.setup(conn)

    # Give the event loop a brief moment to execute the scheduled task
    await asyncio.sleep(0.01)

    assert conn._closing is True
