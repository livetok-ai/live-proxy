import asyncio
import re
from typing import List

import numpy as np
import torch
from PIL.Image import Image

# Check if easyocr is available
try:
    import easyocr
except ImportError:
    easyocr = None

from logger import log_info, log_warn
from providers.vision_model import VisionModel


class OCRProvider(VisionModel):
    _shared_reader = None
    _shared_languages = None

    DEFAULT_SAMPLING_RATE = 5
    DETECTION_EVENT = "texts_detected"

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
        self.last_detected_Boxes = []

        # Parse languages list (e.g. languages="en+es" or "en,fr")
        langs_str = kwargs.get("languages") or kwargs.get("langs") or "en"
        if isinstance(langs_str, str):
            self.languages = [lang.strip() for lang in re.split(r"[+,|]", langs_str) if lang.strip()]
        else:
            self.languages = list(langs_str) if langs_str else ["en"]

        log_info(
            f"OCR provider initialized. draw_detections: {self.draw_detections}, "
            f"sampling_rate: {self.sampling_rate}, languages: {self.languages}"
        )

    @property
    def is_ready(self) -> bool:
        return self.reader is not None

    async def connect(self):
        if easyocr is None:
            log_warn("Cannot connect OCR provider because EasyOCR is not installed.")
            return

        # Ensure setup has been executed
        if OCRProvider._shared_reader is None or OCRProvider._shared_languages != self.languages:
            await OCRProvider.setup(self.languages)

        self.reader = OCRProvider._shared_reader

    def clear_overlay(self):
        self.last_detected_Boxes = []

    async def process_frame(self, image: Image):
        # Convert PIL image to numpy array for EasyOCR
        image_np = np.array(image.convert("RGB"))

        # Run OCR in thread executor to keep asyncio responsive
        loop = asyncio.get_event_loop()
        try:
            results = await loop.run_in_executor(None, lambda: self.reader.readtext(image_np))
        except Exception as e:
            log_warn(f"Error during OCR text detection: {e}")
            results = []

        detected_texts = []
        detected_Boxes = []

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
                detected_Boxes.append({"coords": coords, "label": text_str, "conf": float(conf), "color": color})

        self.last_detected_Boxes = detected_Boxes

        raw = [{"text": b["label"], "coords": b["coords"], "conf": b["conf"]} for b in detected_Boxes]
        self.notify_detections(raw, set(detected_texts))

        objects = []
        for b in detected_Boxes:
            x1, y1, x2, y2 = b["coords"]
            objects.append(
                {
                    "label": b["label"],
                    "left": round(min(1.0, max(0.0, x1 / image.width)), 2),
                    "top": round(min(1.0, max(0.0, y1 / image.height)), 2),
                    "right": round(min(1.0, max(0.0, x2 / image.width)), 2),
                    "bottom": round(min(1.0, max(0.0, y2 / image.height)), 2),
                }
            )
        self.notify_objects(objects)

    async def close(self):
        log_info("Closing OCR provider")
        self.reader = None
