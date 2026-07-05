import asyncio
import os
from typing import AsyncIterator

from av import VideoFrame
from PIL import ImageDraw
from PIL.Image import Image
from ultralytics import YOLO

from logger import log_info
from model import Input, Model, Output
from utils import limit_queue_size, parse_bool, parse_int


class YoloProvider(Model):
    _shared_model = None

    @property
    def supports_video(self) -> bool:
        return True

    @property
    def video_support(self) -> bool:
        return True

    @classmethod
    async def setup(cls):
        if cls._shared_model is not None:
            return

        # Locate yolo11n.pt model. First check local directory, then check examples folder
        model_path = "yolo11n.pt"
        possible_paths = [
            model_path,
            os.path.join(os.path.dirname(__file__), "yolo11n.pt"),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../examples/yolo11n.pt")),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "../../examples/yolo11n.pt")),
        ]

        selected_path = model_path
        for path in possible_paths:
            if os.path.exists(path):
                selected_path = path
                break

        log_info(f"Loading YOLO model from: {selected_path}")
        # Load YOLO in thread pool since constructor or model loading might block
        loop = asyncio.get_event_loop()
        cls._shared_model = await loop.run_in_executor(None, lambda: YOLO(selected_path))

    def __init__(self, name=None, connection=None, **kwargs):
        super().__init__(name=name, connection=connection, **kwargs)
        self.model = kwargs.get("model") or name
        self.last_detections = set()

        # Support draw and sampling parameters only from kwargs
        self.draw_detections = parse_bool(kwargs.get("draw"), False)
        self.sampling_rate = parse_int(kwargs.get("sampling"), 5)

        self.frame_count = 0
        self.last_drawn_boxes = []
        self.output_queue = asyncio.Queue()
        self._processing = False
        log_info(
            f"YOLO provider model: {self.model} draw_detections: {self.draw_detections} sampling_rate: {self.sampling_rate}"
        )

    @property
    def overlay_enabled(self) -> bool:
        return self.draw_detections

    def get_color(self, label: str):
        # Stable, beautiful curated colors
        colors = [
            (255, 75, 75),  # Red
            (75, 123, 255),  # Blue
            (75, 255, 123),  # Green
            (180, 75, 255),  # Purple
            (255, 140, 0),  # Orange
            (0, 206, 209),  # Cyan
            (255, 215, 0),  # Yellow
            (255, 105, 180),  # Pink
            (255, 20, 147),  # Deep Pink
            (0, 250, 154),  # Medium Spring Green
        ]
        h = 0
        for char in label:
            h = (h * 31 + ord(char)) & 0xFFFFFFFF
        return colors[h % len(colors)]

    async def connect(self):
        if YoloProvider._shared_model is None:
            await YoloProvider.setup()

        self.model = YoloProvider._shared_model

    async def send(self, input: Input):
        if not self.model:
            return

        if not isinstance(input, Image):
            return

        self.frame_count += 1
        should_process = (self.frame_count % self.sampling_rate == 1) or (self.sampling_rate <= 1)

        if self.input_enabled and should_process:
            # Drop this frame's inference if the previous one is still processing
            if not self._processing:
                self._processing = True
                asyncio.ensure_future(self._process_frame(input))
        elif not self.input_enabled:
            self.last_drawn_boxes = []
            self.last_detections = set()

        # Send/Overlay detections if output is enabled
        if self.output_enabled:
            if self.overlay_enabled:
                # Copy image to draw boxes on
                drawn_image = input.copy()
                draw = ImageDraw.Draw(drawn_image)

                for box in self.last_drawn_boxes:
                    coords = box["coords"]
                    label = box["label"]
                    conf = box["conf"]
                    color = box["color"]
                    x1, y1, x2, y2 = coords[0], coords[1], coords[2], coords[3]

                    # Draw nice bounding box
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

                    # Text label with confidence
                    text = f"{label} {conf:.2f}"

                    # Draw background box for text label
                    try:
                        bbox = draw.textbbox((x1, y1), text)
                        tw = bbox[2] - bbox[0]
                        th = bbox[3] - bbox[1]
                    except AttributeError:
                        tw, th = draw.textsize(text)

                    text_bg = [x1, max(0, y1 - th - 6), x1 + tw + 6, y1]
                    draw.rectangle(text_bg, fill=color)
                    draw.text((x1 + 3, max(0, y1 - th - 4)), text, fill=(255, 255, 255))
            else:
                drawn_image = input

            new_frame = VideoFrame.from_image(drawn_image)
            if hasattr(input, "pts"):
                new_frame.pts = input.pts
            if hasattr(input, "time_base"):
                new_frame.time_base = input.time_base

            limit_queue_size(self.output_queue, 10)
            self.output_queue.put_nowait(new_frame)

    async def _process_frame(self, input: Image):
        try:
            # Run inference in the default executor (thread pool) to keep asyncio event loop responsive
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(None, lambda: self.model(input, verbose=False))

            detected = set()
            drawn_boxes = []

            if results and len(results) > 0:
                result = results[0]
                if hasattr(result, "boxes") and result.boxes is not None:
                    for b in result.boxes:
                        if b.cls is not None:
                            class_id = int(b.cls)
                            if class_id in self.model.names:
                                label = self.model.names[class_id]
                                detected.add(label)
                                if hasattr(b, "xyxy") and b.xyxy is not None:
                                    coords = b.xyxy[0].tolist()
                                    conf = float(b.conf[0]) if hasattr(b, "conf") and b.conf is not None else 1.0
                                    color = self.get_color(label)
                                    drawn_boxes.append({"coords": coords, "label": label, "conf": conf, "color": color})

            self.last_drawn_boxes = drawn_boxes

            if detected != self.last_detections:
                sorted_detections = sorted(detected)
                log_info(f"YOLO detections changed: {sorted_detections}")
                self.last_detections = detected
                self._emit("detections_changed", sorted_detections)
                self._emit("objects", sorted_detections)
        finally:
            self._processing = False

    async def recv(self) -> AsyncIterator[Output]:
        while True:
            try:
                frame = await self.output_queue.get()
                yield frame
            except asyncio.CancelledError:
                break

    async def close(self):
        log_info("Closing YOLO provider")
        self.model = None
