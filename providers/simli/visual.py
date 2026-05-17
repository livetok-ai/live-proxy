import asyncio
import contextlib
import os
import sys
import time
from typing import AsyncGenerator, AsyncIterator

import numpy as np
from av import AudioFrame, AudioResampler
from dotenv import load_dotenv
from simli import SimliClient, SimliConfig

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from logger import log_info
from model import Input, Model, Output

SAMPLE_RATE = 16000
AUDIO_PTIME = 0.02


class Simli(Model):
    def __init__(self):
        super().__init__()
        self.api_key = None
        self.face_id = None
        self.client = None
        self.connected = False
        self.last_sent = time.time()
        self.send_resampler = AudioResampler(
            format="s16",
            layout="mono",
            rate=16000,
        )
        self.recv_resampler = AudioResampler(
            format="s16",
            layout="mono",
            rate=24000,
        )

    async def connect(self, api_key: str = None, face_id: str = None):
        log_info("Simli connect")

        self.api_key = api_key or os.getenv("SIMLI_API_KEY")
        self.face_id = face_id or os.getenv("SIMLI_FACE_ID")

        if not self.api_key:
            raise ValueError("SIMLI_API_KEY is required")
        if not self.face_id:
            raise ValueError("SIMLI_FACE_ID is required")

        log_info(f"Connecting to Simli with face_id: {self.face_id}")

        config = SimliConfig(faceId=self.face_id, maxSessionLength=300, maxIdleTime=60, model="fasttalk")

        self.client = SimliClient(self.api_key, config)
        await self.client.start()
        self.connected = True

        log_info("Connected to Simli")

        # Workaround to fix latency with simli
        await self.send_silence(10)
        await self.clear()

    async def send_audio(self, input: Input):
        if not self.connected or not self.client:
            log_info("Simli client not connected, skipping send")
            return

        try:
            if isinstance(input, AudioFrame):
                for frame in self.send_resampler.resample(input):
                    audio_data = frame.to_ndarray().astype(np.int16).tobytes()
                    # if time.time() - self.last_sent > 1:
                    #     await self.client.sendImmediate(audio_data)
                    # else:
                    await self.client.send(audio_data)
                    self.last_sent = time.time()
            elif isinstance(input, bytes):
                await self.client.send(input)
        except Exception as e:
            log_info(f"Error sending to Simli: {e}")

    async def send_silence(self, duration: float = 0.1875):
        if not self.connected or not self.client:
            return
        await self.client.sendSilence(duration)

    async def recv_audio(self) -> AsyncIterator[AudioFrame]:
        if not self.connected or not self.client:
            return

        try:
            async for frame in self.client.getAudioStreamIterator(targetSampleRate=24000):
                # Use resampler to convert stereo to mono
                for mono_frame in self.recv_resampler.resample(frame):
                    yield mono_frame
        except Exception as e:
            log_info(f"Error receiving audio from Simli: {e}")

    async def recv_video(self) -> AsyncIterator[Output]:
        if not self.connected or not self.client:
            return

        try:
            async for frame in self.client.getVideoStreamIterator(targetFormat="rgb24"):
                yield frame
        except Exception as e:
            log_info(f"Error receiving video from Simli: {e}")

    async def clear(self):
        await self.client.clearBuffer()

    async def close(self):
        log_info("Closing Simli connection")
        if self.client:
            await self.client.stop()
            self.client = None
        self.connected = False


@contextlib.asynccontextmanager
async def connect_simli(api_key: str = None, face_id: str = None) -> AsyncGenerator[Simli, None]:
    simli = Simli()
    await simli.connect(api_key, face_id)
    try:
        yield simli
    finally:
        await simli.close()


async def main():
    import numpy as np
    import logging

    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    print("Starting Simli test...")

    running = True

    async def send_audio_task(simli: Simli):
        """Send 20ms chunks of random audio until stopped."""
        samples_per_chunk = int(SAMPLE_RATE * AUDIO_PTIME)  # 320 samples for 20ms at 16kHz
        chunk_count = 0

        print("Starting audio sender...")
        while running:
            random_audio = np.random.randint(-32768, 32767, samples_per_chunk, dtype=np.int16)
            audio_bytes = random_audio.tobytes()
            await simli.send_audio(audio_bytes)
            chunk_count += 1
            if chunk_count % 50 == 0:  # Log every second (50 * 20ms = 1000ms)
                print(f"Sent {chunk_count} audio chunks ({chunk_count * AUDIO_PTIME:.1f}s)")
            await asyncio.sleep(AUDIO_PTIME)

        print(f"Audio sender stopped after {chunk_count} chunks")

    async def recv_video_task(simli: Simli):
        """Receive and print video frame resolutions."""
        nonlocal running
        frame_count = 0
        max_frames = 150

        print("Starting video receiver...")
        async for video_frame in simli.recv_video():
            frame_count += 1
            print(f"Frame {frame_count}: {video_frame.shape}")

            if frame_count >= max_frames:
                print(f"Received {max_frames} frames, stopping...")
                running = False
                break

        print(f"Video receiver stopped after {frame_count} frames")

    async with connect_simli() as simli:
        print("Connected to Simli")

        # Send initial silence to start the avatar
        print("Sending 1 second of silence to initialize...")
        await simli.send_silence(1)

        # Run audio sender and video receiver in parallel
        await asyncio.gather(send_audio_task(simli), recv_video_task(simli))


if __name__ == "__main__":
    asyncio.run(main())
