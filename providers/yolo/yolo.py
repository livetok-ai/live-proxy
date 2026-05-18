import asyncio
import os
from typing import AsyncIterator
from PIL.Image import Image
from PIL import ImageDraw
from av import VideoFrame
from ultralytics import YOLO

from logger import log_info
from model import Input, Model, Output


class YoloProvider(Model):
    def __init__(self, draw_detections: bool = False, sampling_rate: int = 5):
        super().__init__()
        self.model = None
        self.last_detections = set()
        self.draw_detections = draw_detections
        self.sampling_rate = sampling_rate
        self.frame_count = 0
        self.last_drawn_boxes = []
        self.output_queue = asyncio.Queue()
        self.client_has_video_recv = False

    @property
    def overlay_enabled(self) -> bool:
        return self.draw_detections and self.client_has_video_recv

    def get_color(self, label: str):
        # Stable, beautiful curated colors
        colors = [
            (255, 75, 75),    # Red
            (75, 123, 255),   # Blue
            (75, 255, 123),   # Green
            (180, 75, 255),   # Purple
            (255, 140, 0),    # Orange
            (0, 206, 209),    # Cyan
            (255, 215, 0),    # Yellow
            (255, 105, 180),  # Pink
            (255, 20, 147),   # Deep Pink
            (0, 250, 154),    # Medium Spring Green
        ]
        h = 0
        for char in label:
            h = (h * 31 + ord(char)) & 0xFFFFFFFF
        return colors[h % len(colors)]

    async def connect(
        self,
        model: str,
        system_instructions=None,
        tools=None,
        tool_callback=None,
        voice=None,
        language=None,
        api_key=None,
        **kwargs,
    ):
        log_info(f"Connecting to YOLO provider: {model}")

        # Check if the video track has recv direction from client side
        connection = kwargs.get("connection")
        self.client_has_video_recv = False
        if connection and connection.pc:
            for transceiver in connection.pc.getTransceivers():
                if transceiver.kind == "video":
                    if transceiver.direction in ("sendonly", "sendrecv") or transceiver.currentDirection in ("sendonly", "sendrecv"):
                        self.client_has_video_recv = True
                        break
        log_info(f"YOLO provider overlay_enabled: {self.overlay_enabled} (draw_detections: {self.draw_detections}, client_has_video_recv: {self.client_has_video_recv})")

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
        self.model = await loop.run_in_executor(None, lambda: YOLO(selected_path))

    async def send(self, input: Input):
        if not self.model:
            return

        if not isinstance(input, Image):
            return

        self.frame_count += 1
        should_process = (self.frame_count % self.sampling_rate == 1) or (self.sampling_rate <= 1)

        if should_process:
            # Run inference in the default executor (thread pool) to keep asyncio event loop responsive
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(None, lambda: self.model(input, verbose=False))

            detected = set()
            self.last_drawn_boxes = []

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
                                    self.last_drawn_boxes.append({
                                        "coords": coords,
                                        "label": label,
                                        "conf": conf,
                                        "color": color
                                    })

            if detected != self.last_detections:
                sorted_detections = sorted(list(detected))
                log_info(f"YOLO detections changed: {sorted_detections}")
                self.last_detections = detected
                self._emit("detections_changed", sorted_detections)

        # Overlay detections if enabled
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

            new_frame = VideoFrame.from_image(drawn_image)
            if hasattr(input, "pts"):
                new_frame.pts = input.pts
            if hasattr(input, "time_base"):
                new_frame.time_base = input.time_base

            self.output_queue.put_nowait(new_frame)

    async def recv(self) -> AsyncIterator[Output]:
        if self.overlay_enabled:
            while True:
                try:
                    frame = await self.output_queue.get()
                    yield frame
                except asyncio.CancelledError:
                    break
        else:
            # Return an empty async generator since YoloProvider does not output audio/video stream
            if False:
                yield

    async def close(self):
        log_info("Closing YOLO provider")
        self.model = None

