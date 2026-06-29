import asyncio
import os
import urllib.request
from typing import AsyncIterator

import numpy as np
from av import VideoFrame
from PIL import ImageDraw
from PIL.Image import Image

from logger import log_info
from model import Input, Model, Output
from utils import limit_queue_size, parse_bool, parse_int


class FaceLandmarkerProvider(Model):
    _shared_model_path = None

    @property
    def supports_video(self) -> bool:
        return True

    @property
    def video_support(self) -> bool:
        return True

    @classmethod
    async def setup(cls):
        if cls._shared_model_path is not None and os.path.exists(cls._shared_model_path):
            return

        model_name = "face_landmarker.task"
        possible_paths = [
            model_name,
            os.path.join(os.path.dirname(__file__), model_name),
            os.path.abspath(os.path.join(os.path.dirname(__file__), f"../../../{model_name}")),
            os.path.abspath(os.path.join(os.path.dirname(__file__), f"../../{model_name}")),
        ]

        cls._shared_model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                cls._shared_model_path = path
                break

        if not cls._shared_model_path:
            cls._shared_model_path = os.path.join(os.path.dirname(__file__), model_name)
            log_info(f"Downloading face_landmarker.task to {cls._shared_model_path}")
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

            # Download in an executor to avoid blocking the main event loop
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: urllib.request.urlretrieve(url, cls._shared_model_path))

        log_info(f"Face Landmarker setup complete: {cls._shared_model_path}")

    def __init__(self, name=None, connection=None, **kwargs):
        super().__init__(name=name, connection=connection, **kwargs)
        self.detector = None
        self.last_emotion = None
        self.model = kwargs.get("model") or name

        # Support draw and sampling parameters only from kwargs
        self.draw_detections = parse_bool(kwargs.get("draw"), False)
        self.sampling_rate = parse_int(kwargs.get("sampling"), 5)

        self.frame_count = 0
        self.last_drawn_boxes = []
        self.output_queue = asyncio.Queue()
        self.model_path = None
        log_info(
            f"Face Landmarker provider draw_detections: {self.draw_detections} sampling_rate: {self.sampling_rate}"
        )

    @property
    def overlay_enabled(self) -> bool:
        return self.draw_detections

    def get_color(self, label: str):
        # Premium harmonized color palette based on detected emotion
        colors = {
            "happy": (255, 215, 0),  # Gold / Warm Yellow
            "sad": (75, 123, 255),  # Soft Blue
            "angry": (255, 75, 75),  # Coral Red
            "surprised": (255, 140, 0),  # Vibrant Orange
            "neutral": (180, 180, 180),  # Light Gray
        }
        return colors.get(label, (0, 206, 209))

    async def connect(self):
        if FaceLandmarkerProvider._shared_model_path is None or not os.path.exists(
            FaceLandmarkerProvider._shared_model_path
        ):
            await FaceLandmarkerProvider.setup()

        self.model_path = FaceLandmarkerProvider._shared_model_path

        loop = asyncio.get_event_loop()
        self.detector = await loop.run_in_executor(None, self._load_detector)

    def _load_detector(self):
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
            num_faces=1,
        )
        return vision.FaceLandmarker.create_from_options(options)

    async def send(self, input: Input):
        if not self.detector:
            return

        if not isinstance(input, Image):
            return

        self.frame_count += 1
        should_process = (self.frame_count % self.sampling_rate == 1) or (self.sampling_rate <= 1)

        if self.input_enabled and should_process:
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(None, self._process_frame, input)

            if results:
                primary_emotion, max_score, coords = results
                if primary_emotion != self.last_emotion:
                    log_info(f"Face emotion changed: {primary_emotion} (confidence: {max_score:.2f})")
                    self.last_emotion = primary_emotion
                    self._emit("emotion_changed", primary_emotion)
                    self._emit("sentiment", primary_emotion)

                self.last_drawn_boxes = []
                if coords:
                    color = self.get_color(primary_emotion)
                    self.last_drawn_boxes.append(
                        {"coords": coords, "label": primary_emotion, "conf": max_score, "color": color}
                    )
            else:
                self.last_drawn_boxes = []
        elif not self.input_enabled:
            self.last_drawn_boxes = []
            self.last_emotion = None

        # Send/Overlay detections if output is enabled
        if self.output_enabled:
            if self.overlay_enabled:
                drawn_image = input.copy()
                draw = ImageDraw.Draw(drawn_image)

                for box in self.last_drawn_boxes:
                    coords = box["coords"]
                    label = box["label"]
                    conf = box["conf"]
                    color = box["color"]
                    x1, y1, x2, y2 = coords[0], coords[1], coords[2], coords[3]

                    # Draw elegant bounding box
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

    def _process_frame(self, image: Image):
        import mediapipe as mp

        image_np = np.array(image.convert("RGB"))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)

        result = self.detector.detect(mp_image)

        if result.face_blendshapes and len(result.face_blendshapes) > 0:
            face_blendshape = result.face_blendshapes[0]
            blendshape_dict = {category.category_name: category.score for category in face_blendshape}

            # Premium rule-based emotion mapping
            emotions = {}
            # Happy: mouthSmileLeft & mouthSmileRight
            emotions["happy"] = (
                blendshape_dict.get("mouthSmileLeft", 0.0) + blendshape_dict.get("mouthSmileRight", 0.0)
            ) / 2.0
            # Sad: mouthFrownLeft & mouthFrownRight
            emotions["sad"] = (
                blendshape_dict.get("mouthFrownLeft", 0.0) + blendshape_dict.get("mouthFrownRight", 0.0)
            ) / 2.0
            # Surprised: jawOpen & eyeWideLeft/Right
            emotions["surprised"] = (
                blendshape_dict.get("jawOpen", 0.0)
                + (blendshape_dict.get("eyeWideLeft", 0.0) + blendshape_dict.get("eyeWideRight", 0.0)) / 2.0
            ) / 2.0
            # Angry: browDownLeft & browDownRight
            emotions["angry"] = (
                blendshape_dict.get("browDownLeft", 0.0) + blendshape_dict.get("browDownRight", 0.0)
            ) / 2.0

            primary_emotion = "neutral"
            max_score = 0.0
            for emotion, score in emotions.items():
                if score > max_score:
                    max_score = score
                    primary_emotion = emotion

            if max_score < 0.25:
                primary_emotion = "neutral"
                max_score = 1.0 - sum(emotions.values())

            coords = None
            if result.face_landmarks and len(result.face_landmarks) > 0:
                face_landmarks = result.face_landmarks[0]
                width, height = image.size
                xs = [lm.x for lm in face_landmarks]
                ys = [lm.y for lm in face_landmarks]
                xmin = min(xs) * width
                xmax = max(xs) * width
                ymin = min(ys) * height
                ymax = max(ys) * height

                # Apply padding
                pad_x = (xmax - xmin) * 0.1
                pad_y = (ymax - ymin) * 0.1
                coords = [
                    max(0, xmin - pad_x),
                    max(0, ymin - pad_y),
                    min(width, xmax + pad_x),
                    min(height, ymax + pad_y),
                ]

            return primary_emotion, max_score, coords
        return None

    async def recv(self) -> AsyncIterator[Output]:
        while True:
            try:
                frame = await self.output_queue.get()
                yield frame
            except asyncio.CancelledError:
                break

    async def close(self):
        log_info("Closing Face Landmarker provider")
        self.detector = None
