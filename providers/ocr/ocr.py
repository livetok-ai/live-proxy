import asyncio
import re
from typing import AsyncIterator, List

import numpy as np
import torch
from av import VideoFrame
from PIL import ImageDraw
from PIL.Image import Image

# Check if easyocr is available
try:
    import easyocr
except ImportError:
    easyocr = None

from logger import log_info, log_warn
from model import Input, Model, Output
from utils import limit_queue_size, parse_bool, parse_int


class OCRProvider(Model):
    _shared_reader = None
    _shared_languages = None

    @property
    def supports_video(self) -> bool:
        return True

    @property
    def video_support(self) -> bool:
        return True

    @classmethod
    async def setup(cls, languages: List[str] = None):
        """Pre-initialize the shared EasyOCR Reader."""
        if easyocr is None:
            log_warn("EasyOCR package is not installed. OCR functionality will be unavailable.")
            return

        if languages is None:
            languages = ["en"]

        # If a reader is already loaded for the same languages, reuse it
        if cls._shared_reader is not None and cls._shared_languages == languages:
            return

        log_info(f"Initializing EasyOCR reader for languages: {languages}")

        # Check GPU availability (CUDA)
        gpu_available = torch.cuda.is_available()
        log_info(f"OCR GPU acceleration available: {gpu_available}")

        # Initialize reader in a thread pool to avoid blocking the main event loop
        loop = asyncio.get_event_loop()
        try:
            cls._shared_reader = await loop.run_in_executor(None, lambda: easyocr.Reader(languages, gpu=gpu_available))
            cls._shared_languages = languages
            log_info("EasyOCR reader initialized successfully.")
        except Exception as e:
            log_warn(f"Failed to initialize EasyOCR reader: {e}")

    def __init__(self, name=None, connection=None, **kwargs):
        super().__init__(name=name, connection=connection, **kwargs)
        self.reader = None
        self.last_detections = set()
        self.model = kwargs.get("model") or name

        # Support draw, sampling, and language parameters from kwargs
        self.draw_detections = parse_bool(kwargs.get("draw"), False)
        self.sampling_rate = parse_int(kwargs.get("sampling"), 5)

        # Parse languages list (e.g. languages="en+es" or "en,fr")
        langs_str = kwargs.get("languages") or kwargs.get("langs") or "en"
        if isinstance(langs_str, str):
            self.languages = [l.strip() for l in re.split(r"[+,|]", langs_str) if l.strip()]
        else:
            self.languages = list(langs_str) if langs_str else ["en"]

        self.frame_count = 0
        self.last_drawn_boxes = []
        self.output_queue = asyncio.Queue()
        log_info(
            f"OCR provider initialized. draw_detections: {self.draw_detections}, "
            f"sampling_rate: {self.sampling_rate}, languages: {self.languages}"
        )

    @property
    def overlay_enabled(self) -> bool:
        return self.draw_detections

    def get_color(self, label: str):
        # Stable, beautiful curated HSL-derived colors for drawing boxes
        colors = [
            (255, 75, 75),  # Coral Red
            (75, 123, 255),  # Soft Blue
            (75, 255, 123),  # Emerald Green
            (180, 75, 255),  # Amethyst Purple
            (255, 140, 0),  # Dark Orange
            (0, 206, 209),  # Dark Turquoise
            (255, 215, 0),  # Gold
            (255, 105, 180),  # Hot Pink
            (0, 250, 154),  # Medium Spring Green
            (238, 130, 238),  # Violet
        ]
        h = 0
        for char in label:
            h = (h * 31 + ord(char)) & 0xFFFFFFFF
        return colors[h % len(colors)]

    async def connect(self):
        if easyocr is None:
            log_warn("Cannot connect OCR provider because EasyOCR is not installed.")
            return

        # Ensure setup has been executed
        if OCRProvider._shared_reader is None or OCRProvider._shared_languages != self.languages:
            await OCRProvider.setup(self.languages)

        self.reader = OCRProvider._shared_reader

    async def send(self, input: Input):
        if not self.reader:
            return

        if not isinstance(input, Image):
            return

        self.frame_count += 1
        should_process = (self.frame_count % self.sampling_rate == 1) or (self.sampling_rate <= 1)

        if self.input_enabled and should_process:
            # Convert PIL image to numpy array for EasyOCR
            image_np = np.array(input.convert("RGB"))

            # Run OCR in thread executor to keep asyncio responsive
            loop = asyncio.get_event_loop()
            try:
                results = await loop.run_in_executor(None, lambda: self.reader.readtext(image_np))
            except Exception as e:
                log_warn(f"Error during OCR text detection: {e}")
                results = []

            detected_texts = []
            self.last_drawn_boxes = []

            if results:
                # Sort results by vertical coordinate (y-min) first, then horizontal (x-min)
                # to read naturally like a document layout
                sorted_results = sorted(results, key=lambda r: (min(pt[1] for pt in r[0]), min(pt[0] for pt in r[0])))

                for bbox, text, conf in sorted_results:
                    if not text or not text.strip():
                        continue

                    text_str = text.strip()
                    detected_texts.append(text_str)

                    # Extract coordinates (min/max of the 4 bbox points)
                    xs = [pt[0] for pt in bbox]
                    ys = [pt[1] for pt in bbox]
                    coords = [min(xs), min(ys), max(xs), max(ys)]

                    color = self.get_color(text_str)
                    self.last_drawn_boxes.append(
                        {"coords": coords, "label": text_str, "conf": float(conf), "color": color}
                    )

            # Check if the unique set of detected texts changed
            detected_set = set(detected_texts)
            if detected_set != self.last_detections:
                sorted_unique_texts = sorted(list(detected_set))
                log_info(f"OCR texts changed: {sorted_unique_texts}")
                self.last_detections = detected_set

                # Emit events
                self._emit("texts_changed", sorted_unique_texts)
                self._emit("texts", sorted_unique_texts)
                self._emit("ocr_text", detected_texts)

        elif not self.input_enabled:
            self.last_drawn_boxes = []
            self.last_detections = set()

        # Generate and enqueue the output frame if output is enabled
        if self.output_enabled:
            if self.overlay_enabled and self.last_drawn_boxes:
                # Draw detected text bounding boxes
                drawn_image = input.copy()
                draw = ImageDraw.Draw(drawn_image)

                for box in self.last_drawn_boxes:
                    coords = box["coords"]
                    label = box["label"]
                    conf = box["conf"]
                    color = box["color"]
                    x1, y1, x2, y2 = coords[0], coords[1], coords[2], coords[3]

                    # Draw neat border
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

                    # Prepare text overlay label (truncated if too long to look beautiful)
                    display_label = label if len(label) < 25 else label[:22] + "..."
                    text = f"{display_label} ({conf:.2f})"

                    # Compute background box for the text label
                    try:
                        bbox_text = draw.textbbox((x1, y1), text)
                        tw = bbox_text[2] - bbox_text[0]
                        th = bbox_text[3] - bbox_text[1]
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

    async def recv(self) -> AsyncIterator[Output]:
        while True:
            try:
                frame = await self.output_queue.get()
                yield frame
            except asyncio.CancelledError:
                break

    async def close(self):
        log_info("Closing OCR provider")
        self.reader = None
