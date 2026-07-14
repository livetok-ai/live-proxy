import asyncio
import os
import urllib.request

import numpy as np
from PIL import ImageDraw
from PIL.Image import Image

from logger import log_info
from providers.vision_model import VisionModel


class FaceLandmarkerProvider(VisionModel):
    _shared_model_path = None

    DEFAULT_SAMPLING_RATE = 5
    DETECTION_EVENT = "emotions_detected"

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
        self.model_path = None
        self.last_detected_Boxes = []
        log_info(
            f"Face Landmarker provider draw_detections: {self.draw_detections} sampling_rate: {self.sampling_rate}"
        )

    @property
    def is_ready(self) -> bool:
        return self.detector is not None

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

    def clear_overlay(self):
        self.last_detected_Boxes = []

    async def process_frame(self, image: Image):
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, self._detect_emotion, image)

        if not results:
            self.last_detected_Boxes = []
            return

        primary_emotion, max_score, coords = results

        self.last_detected_Boxes = []
        if coords:
            color = self.get_color(primary_emotion)
            self.last_detected_Boxes.append(
                {"coords": coords, "label": primary_emotion, "conf": max_score, "color": color}
            )

        raw = {"emotion": primary_emotion, "confidence": max_score, "coords": coords}
        self.notify_detections(raw, primary_emotion)

        objects = []
        if coords:
            objects.append(
                {
                    "label": primary_emotion,
                    "left": round(min(1.0, max(0.0, coords[0] / image.width)), 2),
                    "top": round(min(1.0, max(0.0, coords[1] / image.height)), 2),
                    "right": round(min(1.0, max(0.0, coords[2] / image.width)), 2),
                    "bottom": round(min(1.0, max(0.0, coords[3] / image.height)), 2),
                }
            )
        self.notify_objects(objects)

    def _detect_emotion(self, image: Image):
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

    async def close(self):
        log_info("Closing Face Landmarker provider")
        self.detector = None
