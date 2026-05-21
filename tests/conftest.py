import sys
from unittest.mock import MagicMock

# Mock cv2 and ultralytics so tests can run without having OpenCV installed on macOS ARM
sys.modules["cv2"] = MagicMock()
sys.modules["ultralytics"] = MagicMock()
