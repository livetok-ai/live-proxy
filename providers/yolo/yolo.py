import asyncio
import os

from PIL.Image import Image
from ultralytics import YOLO

from logger import log_info
from providers.vision_model import VisionModel


class YoloProvider(VisionModel):
    _shared_model = None

    DEFAULT_SAMPLING_RATE = 5

    @classmethod
    async def setup(cls):
        if cls._shared_model is not None:
            return

        # Locate yolo11n.pt model. First check local directory, then check examples folder
        model_path = "yolo11n.pt"
        possible_paths = [
            os.path.join(os.path.dirname(__file__), model_path),
            model_path,
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
        self.model_name = kwargs.get("model") or name
        self.model = None
        self.last_detected_Boxes = []
        log_info(f"YOLO provider model: {self.model_name} sampling_rate: {self.sampling_rate}")

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    async def connect(self):
        if YoloProvider._shared_model is None:
            await YoloProvider.setup()

        self.model = YoloProvider._shared_model

    def clear_overlay(self):
        self.last_detected_Boxes = []

    async def process_frame(self, image: Image):
        # Run inference in the default executor, serialized against other connections
        # sharing the same model instance (see Model.run_shared_inference).
        results = await self.run_shared_inference(lambda: self.model(image, verbose=False))

        detected = set()
        detected_Boxes = []

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
                                coords = [round(c, 2) for c in b.xyxy[0].tolist()]
                                conf = float(b.conf[0]) if hasattr(b, "conf") and b.conf is not None else 1.0
                                color = self.get_color(label)
                                detected_Boxes.append({"coords": coords, "label": label, "conf": conf, "color": color})

        self.last_detected_Boxes = detected_Boxes

        raw = [{"label": b["label"], "coords": b["coords"], "conf": b["conf"]} for b in detected_Boxes]
        self.notify_detections(raw, detected)

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
        log_info("Closing YOLO provider")
        self.model = None
