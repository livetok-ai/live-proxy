import asyncio
import os
import sys
from typing import AsyncIterator

from PIL import Image

from logger import log_info
from model import Input, Model, Output


class InceptionProvider(Model):
    _shared_mtcnn = None
    _shared_resnet = None
    _shared_device = None

    @classmethod
    async def setup(cls):
        if cls._shared_resnet is not None:
            return

        import torch
        from facenet_pytorch import MTCNN, InceptionResnetV1

        cls._shared_device = torch.device("cpu")
        log_info(f"Loading FaceNet MTCNN and InceptionResnetV1 on device: {cls._shared_device}")

        loop = asyncio.get_event_loop()
        def _load():
            mtcnn = MTCNN(keep_all=False, device=cls._shared_device)
            resnet = InceptionResnetV1(pretrained="vggface2").eval().to(cls._shared_device)
            return mtcnn, resnet

        cls._shared_mtcnn, cls._shared_resnet = await loop.run_in_executor(None, _load)
        log_info("Inception FaceNet models setup and loaded successfully")

    def __init__(self, sampling_rate: int = 150, **kwargs):
        """Inception (FaceNet) Provider for extracting face embeddings.

        Args:
            sampling_rate: Number of frames to wait before processing the next frame.
                           For ~30 FPS video, 150 frames = once every 5 seconds.
        """
        super().__init__()
        self.mtcnn = None
        self.resnet = None
        self.device = None

        # Support sampling rate from kwargs or default
        self.sampling_rate = kwargs.get("sampling", sampling_rate)
        if isinstance(self.sampling_rate, (int, str)):
            self.sampling_rate = int(self.sampling_rate)

        self.frame_count = 0
        self.last_process_time = 0.0
        # By default, we let YoloProvider control input_enabled dynamically.
        # But we can default it to True in class __init__ if needed.
        self.input_enabled = True

    async def connect(self, name: str = None, connection=None, model: str = None):
        model = name or model
        log_info(f"Connecting to Inception FaceNet provider: {model}")

        if InceptionProvider._shared_resnet is None:
            await InceptionProvider.setup()

        self.device = InceptionProvider._shared_device
        self.mtcnn = InceptionProvider._shared_mtcnn
        self.resnet = InceptionProvider._shared_resnet

    async def send(self, input: Input):
        if not self.resnet:
            return

        if not isinstance(input, Image.Image):
            return

        self.frame_count += 1
        should_process = (self.frame_count % self.sampling_rate == 1) or (self.sampling_rate <= 1)

        if self.input_enabled and should_process:
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(None, self._process_frame, input)

            if embedding is not None:
                embedding_list = embedding.tolist()
                log_info(f"Inception extracted face embedding successfully (size: {len(embedding_list)})")
                self._emit("face_embeddings", embedding_list)

    def _process_frame(self, image: Image.Image):
        import torch

        # MTCNN detects face, crops, aligns and returns PyTorch tensor of shape [3, 160, 160]
        face_tensor = self.mtcnn(image)
        if face_tensor is not None:
            # Resnet expects batch dimension: [1, 3, 160, 160]
            with torch.no_grad():
                embedding_tensor = self.resnet(face_tensor.unsqueeze(0))
                # Returns 512-dimensional embedding
                return embedding_tensor.squeeze(0).cpu().numpy()
        return None

    async def recv(self) -> AsyncIterator[Output]:
        # Return empty generator because this model only extracts embeddings and emits events,
        # it does not modify or produce outgoing video frames
        if False:
            yield

    async def close(self):
        log_info("Closing Inception provider")
        self.mtcnn = None
        self.resnet = None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inception.py <path_to_image_file>")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' does not exist.")
        sys.exit(1)

    print(f"Loading image from: {image_path}")
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error opening image: {e}")
        sys.exit(1)

    print("Loading PyTorch & facenet-pytorch...")
    import torch
    from facenet_pytorch import MTCNN, InceptionResnetV1

    # Select device (using CPU due to MPS adaptive average pooling bugs)
    device = torch.device("cpu")
    print(f"Using device: {device}")

    print("Initializing MTCNN Face Detector...")
    mtcnn = MTCNN(keep_all=False, device=device)

    print("Initializing InceptionResnetV1 Face Embedder (pretrained='vggface2')...")
    resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    print("Detecting and cropping face...")
    face_tensor = mtcnn(img)

    if face_tensor is None:
        print("❌ No face detected in the image.")
        sys.exit(1)

    print("Face detected! Extracting embedding representation...")
    with torch.no_grad():
        embedding_tensor = resnet(face_tensor.unsqueeze(0))
        embedding = embedding_tensor.squeeze(0).cpu().numpy()

    print("\n✅ Embedding extracted successfully!")
    print(f"Vector Dimensions: {len(embedding)}")
    print(f"Data Type: {embedding.dtype}")
    print("\nEmbedding Vector:")
    print(embedding.tolist())
