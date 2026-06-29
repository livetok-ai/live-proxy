import asyncio
from typing import AsyncIterator

from av import VideoFrame
from PIL import Image as PILImage
from PIL import ImageDraw
from PIL.Image import Image
from ultralytics import SAM

from logger import log_info
from model import Input, Model, Output
from utils import limit_queue_size, parse_bool, parse_int


class SamProvider(Model):
    _shared_model = None
    _loaded_model_version = None

    @property
    def supports_video(self) -> bool:
        return True

    @property
    def video_support(self) -> bool:
        return True

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

        # Support draw and sampling parameters only from kwargs
        self.draw_detections = parse_bool(kwargs.get("draw"), False)
        self.sampling_rate = parse_int(kwargs.get("sampling"), 5)

        self.device = kwargs.get("device", None)
        self.frame_count = 0
        self.last_masks = []
        self.output_queue = asyncio.Queue()
        log_info(
            f"SAM provider version: {self.model_version} draw_detections: {self.draw_detections} sampling_rate: {self.sampling_rate}"
        )

    @property
    def overlay_enabled(self) -> bool:
        return self.draw_detections

    def get_color(self, label: str):
        # Stable, beautiful curated colors for segmentation overlays
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

    async def send(self, input: Input):
        if not self.model:
            return

        if not isinstance(input, Image):
            return

        self.frame_count += 1
        should_process = (self.frame_count % self.sampling_rate == 1) or (self.sampling_rate <= 1)

        if self.input_enabled and should_process:
            # Run inference in the default executor (thread pool) to keep asyncio event loop responsive
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(None, lambda: self.model(input, device=self.device, verbose=False))

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

            current_labels = [m["label"] for m in detected_masks]
            last_labels = [m["label"] for m in self.last_masks]
            if current_labels != last_labels:
                log_info(f"SAM3 objects changed: {current_labels}")
                self.last_masks = detected_masks
                self._emit("segmentations_changed", current_labels)
                self._emit("objects", current_labels)
            else:
                # Update masks to preserve smooth spatial mapping if desired
                self.last_masks = detected_masks

        elif not self.input_enabled:
            self.last_masks = []

        # Send/Overlay detections if output is enabled
        if self.output_enabled:
            if self.overlay_enabled:
                # Copy input to draw on
                drawn_image = input.copy()

                # Create semi-transparent overlay mask layer
                mask_layer = PILImage.new("RGBA", drawn_image.size, (0, 0, 0, 0))
                draw_mask = ImageDraw.Draw(mask_layer)

                for mask in self.last_masks:
                    coords = mask["coords"]
                    color = mask["color"]
                    label = mask["label"]
                    center_x, center_y = mask["center"]

                    polygon_points = [tuple(p) for p in coords]
                    if len(polygon_points) >= 3:
                        # Draw a nice translucent mask fill + outline
                        fill_color = color + (80,)
                        outline_color = color + (200,)
                        draw_mask.polygon(polygon_points, fill=fill_color, outline=outline_color, width=2)

                        # Text label centered inside the segmented object
                        text = label
                        try:
                            bbox = draw_mask.textbbox((center_x, center_y), text)
                            tw = bbox[2] - bbox[0]
                            th = bbox[3] - bbox[1]
                        except AttributeError:
                            tw, th = draw_mask.textsize(text)

                        # Draw text background and text centered
                        text_x = max(0, min(drawn_image.width - tw - 6, center_x - tw / 2.0))
                        text_y = max(0, min(drawn_image.height - th - 6, center_y - th / 2.0))

                        text_bg = [text_x, text_y, text_x + tw + 6, text_y + th + 4]
                        draw_mask.rectangle(text_bg, fill=color + (255,))
                        draw_mask.text((text_x + 3, text_y + 1), text, fill=(255, 255, 255))

                # Composite the mask overlay with the original frame
                final_image = PILImage.alpha_composite(drawn_image.convert("RGBA"), mask_layer).convert("RGB")
            else:
                final_image = input

            new_frame = VideoFrame.from_image(final_image)
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
        log_info("Closing SAM provider")
        self.model = None
