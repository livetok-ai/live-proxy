import asyncio

from PIL import Image as PILImage
from PIL import ImageDraw
from PIL.Image import Image
from ultralytics import SAM

from logger import log_info
from providers.vision_model import VisionModel, draw_centered_label


class SamProvider(VisionModel):
    _shared_model = None
    _loaded_model_version = None

    DEFAULT_SAMPLING_RATE = 5

    @classmethod
    async def setup(cls, model_version: str = "sam2.1_t.pt"):
        if cls._shared_model is not None and cls._loaded_model_version == model_version:
            return f"already loaded version: {model_version}"

        # Load SAM in a thread pool since constructor or model loading might block
        loop = asyncio.get_event_loop()
        cls._shared_model = await loop.run_in_executor(None, lambda: SAM(model_version))
        cls._loaded_model_version = model_version
        return f"loaded version: {model_version}"

    def __init__(self, name=None, connection=None, **kwargs):
        super().__init__(name=name, connection=connection, **kwargs)
        self.model = None
        self.model_version = kwargs.get("version", "sam2.1_t.pt")
        self.device = kwargs.get("device", None)
        self.last_masks = []
        log_info(
            f"SAM provider version: {self.model_version} draw_detections: {self.draw_detections} "
            f"sampling_rate: {self.sampling_rate}"
        )

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    async def connect(self):
        # Setup the shared model
        await SamProvider.setup(self.model_version)
        self.model = SamProvider._shared_model

        # Auto-detect optimal device if none is specified
        if self.device is None:
            import torch

            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"

    def clear_overlay(self):
        self.last_masks = []

    def draw_overlay(self, image: Image) -> Image:
        drawn_image = image.copy()

        # Create semi-transparent overlay mask layer
        mask_layer = PILImage.new("RGBA", drawn_image.size, (0, 0, 0, 0))
        draw_mask = ImageDraw.Draw(mask_layer)

        for mask in self.last_masks:
            color = mask["color"]
            label = mask["label"]

            polygon_points = [tuple(p) for p in mask["coords"]]
            if len(polygon_points) >= 3:
                # Draw a nice translucent mask fill + outline
                fill_color = color + (80,)
                outline_color = color + (200,)
                draw_mask.polygon(polygon_points, fill=fill_color, outline=outline_color, width=2)

                # Text label centered inside the segmented object
                draw_centered_label(draw_mask, mask["center"], label, color + (255,), drawn_image.size)

        # Composite the mask overlay with the original frame
        return PILImage.alpha_composite(drawn_image.convert("RGBA"), mask_layer).convert("RGB")

    async def process_frame(self, image: Image):
        # Run inference in the default executor (thread pool) to keep asyncio event loop responsive
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, lambda: self.model(image, device=self.device, verbose=False))

        detected_masks = []
        if results and len(results) > 0:
            result = results[0]
            if hasattr(result, "masks") and result.masks is not None:
                # result.masks.xy contains a list of polygons/contours as numpy arrays
                for idx, contour in enumerate(result.masks.xy):
                    coords = contour.tolist()  # List of [x, y] coordinates
                    if len(coords) > 0:
                        xs = [pt[0] for pt in coords]
                        ys = [pt[1] for pt in coords]
                        xmin, xmax = min(xs), max(xs)
                        ymin, ymax = min(ys), max(ys)
                        width = xmax - xmin
                        height = ymax - ymin
                        area = width * height
                        center_x = xmin + width / 2.0
                        center_y = ymin + height / 2.0

                        label = f"object_{idx + 1}"
                        color = self.get_color(label)
                        detected_masks.append(
                            {
                                "id": idx + 1,
                                "label": label,
                                "coords": coords,
                                "area": area,
                                "center": (center_x, center_y),
                                "color": color,
                            }
                        )

        self.last_masks = detected_masks

        raw = [
            {"label": m["label"], "coords": m["coords"], "area": m["area"], "center": m["center"]}
            for m in detected_masks
        ]
        current_labels = {m["label"] for m in detected_masks}
        self.notify_detections(raw, current_labels)

    async def close(self):
        log_info("Closing SAM provider")
        self.model = None
