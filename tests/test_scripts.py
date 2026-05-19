import os
from unittest.mock import patch

import pytest

import script_manager
from connection import Connection
from providers.face_sentiment.face_sentiment import FaceSentimentProvider
from providers.yolo.yolo import YoloProvider


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


class DummyFaceSentimentModel(FaceSentimentProvider):
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
    face_model = DummyFaceSentimentModel()
    conn.models = [yolo_model, face_model]

    assert conn.get_model("yolo") is yolo_model
    assert conn.get_model("face_sentiment") is face_model
    assert conn.get_model("nonexistent") is None


@pytest.mark.asyncio
async def test_yolo_face_sentiment_enable_disable():
    """Test enable/disable methods and input/output flags on YOLO/FaceSentiment providers."""
    yolo = YoloProvider()
    face = FaceSentimentProvider()

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
    landmark = DummyFaceSentimentModel()

    conn.models = [yolo, landmark]

    # Load scripts and find counter module
    script_manager.load_all_scripts()
    counter_script = next(m for m in script_manager.LOADED_SCRIPTS if m.__name__ == "counter")

    # Run setup
    counter_script.setup(conn)

    # Check that handlers were registered
    assert "objects" in yolo._handlers
    assert "sentiment" in landmark._handlers

    # Check initial default state set by setup: YOLO active, Landmark inactive
    assert yolo.input_enabled is True
    assert yolo.output_enabled is True
    assert landmark.input_enabled is False
    assert landmark.output_enabled is False

    # 1. Trigger objects detected (no person)
    yolo.trigger("objects", ["car", "dog"])
    assert yolo.output_enabled is True
    assert landmark.input_enabled is False
    assert landmark.output_enabled is False
    assert conn.object_counts == {"car": 1, "dog": 1}

    # 2. Trigger objects detected (with person)
    yolo.trigger("objects", ["person", "chair"])
    assert yolo.output_enabled is False
    assert landmark.input_enabled is True
    assert landmark.output_enabled is True
    assert conn.object_counts == {"car": 1, "dog": 1, "person": 1, "chair": 1}

    # 3. Trigger sentiment detected
    landmark.trigger("sentiment", "happy")
    landmark.trigger("sentiment", "happy")
    landmark.trigger("sentiment", "neutral")
    assert conn.sentiment_counts == {"happy": 2, "neutral": 1}

    # 4. Run teardown
    with patch("logger.log_info") as mock_log_info:
        counter_script.teardown(conn)

        # Verify log output called for final counts
        log_messages = [call[0][0] for call in mock_log_info.call_args_list]
        assert any("Final object counts:" in msg for msg in log_messages)
        assert any("Final sentiment counts:" in msg for msg in log_messages)

    # Check event handlers were cleaned up
    assert "objects" not in yolo._handlers
    assert "sentiment" not in landmark._handlers


@pytest.mark.asyncio
async def test_js_counter_script_logic():
    """Test JS setup, objects detected (person vs other), sentiment detected, and teardown."""
    conn = Connection()

    yolo = DummyYoloModel()
    landmark = DummyFaceSentimentModel()

    conn.models = [yolo, landmark]

    # Load scripts and find counter JS wrapper
    script_manager.load_all_scripts()
    counter_script = next(m for m in script_manager.LOADED_SCRIPTS if m.__name__ == "counter.js")

    # Run setup
    counter_script.setup(conn)

    # Check that handlers were registered
    assert "objects" in yolo._handlers
    assert "sentiment" in landmark._handlers

    # Check initial default state set by setup: YOLO active, Landmark inactive
    assert yolo.input_enabled is True
    assert yolo.output_enabled is True
    assert landmark.input_enabled is False
    assert landmark.output_enabled is False

    # 1. Trigger objects detected (no person)
    yolo.trigger("objects", ["car", "dog"])
    assert yolo.output_enabled is True
    assert landmark.input_enabled is False
    assert landmark.output_enabled is False

    # 2. Trigger objects detected (with person)
    yolo.trigger("objects", ["person", "chair"])
    assert yolo.output_enabled is False
    assert landmark.input_enabled is True
    assert landmark.output_enabled is True

    # 3. Trigger sentiment detected
    landmark.trigger("sentiment", "happy")
    landmark.trigger("sentiment", "happy")
    landmark.trigger("sentiment", "neutral")

    # 4. Run teardown
    with patch("logger.log_info") as mock_log_info:
        counter_script.teardown(conn)

        # Verify log output called for final counts
        log_messages = [call[0][0] for call in mock_log_info.call_args_list]
        assert any("Final object counts:" in msg for msg in log_messages)
        assert any("Final sentiment counts:" in msg for msg in log_messages)

    # Check event handlers were cleaned up
    assert "objects" not in yolo._handlers
    assert "sentiment" not in landmark._handlers
