import json
import os
import re

from google import genai
from google.genai import types as genai_types
from PIL import ImageDraw
from PIL.Image import Image

from logger import log_info
from providers.vision_model import VisionModel, draw_box_with_label

DEFAULT_MODEL = "gemini-robotics-er-1.6-preview"
DEFAULT_PROMPT = (
    "Detect the objects in the image relevant to a robot manipulating the scene. "
    "Output a JSON array where each entry is an object with keys "
    "'box_2d' (a list [ymin, xmin, ymax, xmax] normalized to 0-1000) and 'label' "
    "(a short name for the object). Return only the JSON array."
)


class GeminiRoboticsProvider(VisionModel):
    def __init__(self, name=None, connection=None, **kwargs):
        super().__init__(name=name, connection=connection, **kwargs)
        self.model = kwargs.get("model") or name
        if self.model:
            # `name` may still carry the raw "provider[param=value,...]" suffix
            # when instantiated directly from the model string (see connection.py).
            self.model = re.sub(r"\[.*\]\s*$", "", self.model).strip()
        if not self.model or self.model in ("gemini_robotics", "gemini-robotics", "robotics"):
            self.model = DEFAULT_MODEL

        self.prompt = kwargs.get("prompt") or DEFAULT_PROMPT
        self.api_key = (
            kwargs.get("api_key") or (connection.api_key if connection else None) or os.getenv("GOOGLE_API_KEY")
        )

        self.client = None
        self.last_drawn_boxes = []

        log_info(
            f"Gemini Robotics provider model: {self.model} draw_detections: {self.draw_detections} "
            f"sampling_rate: {self.sampling_rate}"
        )

    @property
    def is_ready(self) -> bool:
        return self.client is not None

    async def connect(self):
        self.client = genai.Client(api_key=self.api_key)

    def clear_overlay(self):
        self.last_drawn_boxes = []

    def draw_overlay(self, image: Image) -> Image:
        drawn_image = image.copy()
        draw = ImageDraw.Draw(drawn_image)

        for box in self.last_drawn_boxes:
            draw_box_with_label(draw, box["coords"], box["label"], box["color"])

        return drawn_image

    async def process_frame(self, image: Image):
        width, height = image.width, image.height
        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=[self.prompt, image],
            config=genai_types.GenerateContentConfig(
                temperature=0.5,
                thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
            ),
        )

        detections = self._parse_detections(response.text or "", width, height)

        drawn_boxes = []
        labels = set()
        for detection in detections:
            labels.add(detection["label"])
            drawn_boxes.append(
                {
                    "coords": detection["coords"],
                    "label": detection["label"],
                    "color": self.get_color(detection["label"]),
                }
            )

        self.last_drawn_boxes = drawn_boxes

        self.notify_detections(detections, labels)

    def _parse_detections(self, text: str, width: int, height: int):
        cleaned = text.strip()
        fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
        if fenced:
            cleaned = fenced.group(1)

        try:
            items = json.loads(cleaned)
        except (json.JSONDecodeError, TypeError):
            return []

        if not isinstance(items, list):
            return []

        detections = []
        for item in items:
            if not isinstance(item, dict):
                continue

            box = item.get("box_2d")
            if not isinstance(box, list) or len(box) != 4:
                continue

            label = str(item.get("label", "object"))
            ymin, xmin, ymax, xmax = box
            x1 = xmin / 1000 * width
            y1 = ymin / 1000 * height
            x2 = xmax / 1000 * width
            y2 = ymax / 1000 * height
            detections.append({"coords": [x1, y1, x2, y2], "label": label, "box_2d": box})

        return detections

    async def close(self):
        log_info("Closing Gemini Robotics provider")
        self.client = None
