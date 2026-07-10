import asyncio
import os

from PIL import ImageDraw
from PIL.Image import Image
from ultralytics import YOLO

from logger import log_info
from providers.vision_model import VisionModel, draw_box_with_label


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
        self.model_name = kwargs.get("model") or name
        self.model = None
        self.last_drawn_boxes = []
        log_info(
            f"YOLO provider model: {self.model_name} draw_detections: {self.draw_detections} "
            f"sampling_rate: {self.sampling_rate}"
        )

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    async def connect(self):
        if YoloProvider._shared_model is None:
            await YoloProvider.setup()

        self.model = YoloProvider._shared_model

    def clear_overlay(self):
        self.last_drawn_boxes = []

    def draw_overlay(self, image: Image) -> Image:
        drawn_image = image.copy()
        draw = ImageDraw.Draw(drawn_image)

        for box in self.last_drawn_boxes:
            text = f'{box["label"]} {box["conf"]:.2f}'
            draw_box_with_label(draw, box["coords"], text, box["color"])

        return drawn_image

    async def process_frame(self, image: Image):
        # Run inference in the default executor (thread pool) to keep asyncio event loop responsive
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, lambda: self.model(image, verbose=False))

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

        raw = [{"label": b["label"], "coords": b["coords"], "conf": b["conf"]} for b in drawn_boxes]
        self.notify_detections(raw, detected)

    async def close(self):
        log_info("Closing YOLO provider")
        self.model = None
