import asyncio
import json
import time

from PIL.Image import Image
from ultralytics import SAM

from logger import log_debug, log_info
from model import Input
from providers.vision_model import VisionModel


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
        # Normalized (0-1) point prompt set via a "pointer" control message from the
        # client (see send()). SAM is a promptable model: calling it with no prompt at
        # all makes ultralytics run its dense "segment everything" grid search, which
        # takes tens of seconds per frame and is not viable for live video. Defaulting
        # to the frame center keeps inference fast (a single-point forward pass) even
        # before the client sends a click.
        self.point = None
        log_info(f"SAM provider version: {self.model_version} sampling_rate: {self.sampling_rate}")

    async def send(self, input: Input):
        if isinstance(input, str):
            self._handle_control_message(input)
            return
        # Call VisionModel's original, un-instrumented send() (via the __wrapped__
        # reference functools.wraps leaves behind): Model.__init_subclass__ already
        # wrapped VisionModel.send to update self.stats, and since this method is
        # itself wrapped the same way, going through super().send() here would count
        # every video frame twice.
        await VisionModel.send.__wrapped__(self, input)

    def _handle_control_message(self, message: str):
        try:
            data = json.loads(message)
        except (TypeError, ValueError):
            return
        if not isinstance(data, dict):
            return

        msg_type = data.get("type")
        if msg_type == "pointer":
            x, y = data.get("x"), data.get("y")
            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                self.point = (min(max(float(x), 0.0), 1.0), min(max(float(y), 0.0), 1.0))
        elif msg_type == "pointer_clear":
            self.point = None

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    async def load(self):
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

        log_debug(f"SAM loaded with device {self.device}")

    def clear_overlay(self):
        self.last_masks = []

    async def process_frame(self, image: Image):
        # Snapshot self.model rather than reading it inside the executor lambda:
        # close() can null out self.model concurrently while this frame is still
        # in flight (send() schedules processing via asyncio.ensure_future without
        # tracking/cancelling the task), which previously raced with the executor
        # thread's read and blew up with "'NoneType' object is not callable".
        model = self.model
        if model is None:
            return

        # SAM requires a prompt to run its fast, promptable path (see send()/self.point
        # above); default to the frame center until the client clicks somewhere.
        px, py = self.point if self.point is not None else (0.5, 0.5)
        point = [[px * image.width, py * image.height]]

        log_debug(f"SAM frame #{self.frame_count} processing started", context=self._log_context)
        start = time.monotonic()

        # Run inference in the default executor, serialized against other connections
        # sharing the same model instance (see Model.run_shared_inference).
        results = await self.run_shared_inference(
            lambda: model(image, device=self.device, points=point, labels=[1], verbose=False)
        )

        elapsed_ms = (time.monotonic() - start) * 1000
        log_debug(
            f"SAM frame #{self.frame_count} processing finished in {elapsed_ms:.1f}ms",
            context=self._log_context,
        )

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

                        # SAM only segments, it never classifies, so there is no real
                        # object name to attach here. Leave the label empty (rather than
                        # a synthetic "object_N") so the UI doesn't render a fake name,
                        # while still keying the color off a stable per-mask id so masks
                        # stay visually distinguishable across frames.
                        mask_id = idx + 1
                        color = self.get_color(str(mask_id))
                        detected_masks.append(
                            {
                                "id": mask_id,
                                "label": "",
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
        # No real label to key change-detection off of (see comment above), so use the
        # set of mask ids instead of m["label"] to detect when the segmentation changed.
        current_mask_ids = {str(m["id"]) for m in detected_masks}
        self.notify_detections(raw, current_mask_ids)

        objects = []
        for m in detected_masks:
            xs = [pt[0] for pt in m["coords"]]
            ys = [pt[1] for pt in m["coords"]]
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
            points = [
                [
                    round(min(1.0, max(0.0, pt[0] / image.width)), 4),
                    round(min(1.0, max(0.0, pt[1] / image.height)), 4),
                ]
                for pt in m["coords"]
            ]
            objects.append(
                {
                    "label": m["label"],
                    "id": m["id"],
                    "left": round(min(1.0, max(0.0, xmin / image.width)), 2),
                    "top": round(min(1.0, max(0.0, ymin / image.height)), 2),
                    "right": round(min(1.0, max(0.0, xmax / image.width)), 2),
                    "bottom": round(min(1.0, max(0.0, ymax / image.height)), 2),
                    "points": points,
                }
            )
        self.notify_objects(objects)

    async def close(self):
        log_info("Closing SAM provider")
        self.model = None
