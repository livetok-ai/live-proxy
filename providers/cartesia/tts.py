import asyncio
import os
import sys
import uuid
from typing import AsyncIterator

from av import AudioFrame
from cartesia import AsyncCartesia

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from logger import log_info
from model import Input, Model, Output
from providers.cartesia.utils import aggreagate_sentences, parse_speakeable_text

# Cartesia uses 22050 Hz sample rate for pcm_f32le format
SAMPLE_RATE = 22050
AUDIO_PTIME = 0.02  # 20ms frames
BYTES_PER_SAMPLE = 4  # 32-bit float = 4 bytes


class CartesiaTTS(Model):
    """
    Cartesia TTS model implementation using WebSocket streaming API.

    This model only supports text-to-speech (TTS) functionality.
    Audio input is ignored, and only text input is converted to speech.
    """

    def __init__(self, voice_id: str = None, model_id: str = "sonic-3", api_key: str = None):
        super().__init__()
        self.voice_id = voice_id or "e07c00bc-4134-4eae-9ea4-1a55fb45746b"
        self.model_id = model_id
        self.api_key = api_key or os.getenv("CARTESIA_API_KEY")

        if not self.api_key:
            raise ValueError("Cartesia API key is required. Set CARTESIA_API_KEY environment variable.")

        self.client = None
        self.ws = None
        self._input_queue = asyncio.Queue()  # For text inputs to be processed
        self._output_queue = asyncio.Queue()  # For converted AudioFrames
        self._processor_task = None  # Task that processes input queue
        self._closed = False
        self._receiving = False
        self._context_id = str(uuid.uuid4())
        self._pending_text = ""

    @property
    def connected(self):
        return self.client is not None

    async def connect(self):
        """Establish connection to Cartesia TTS service."""
        log_info(f"Connecting to Cartesia TTS with model={self.model_id}, voice={self.voice_id}")

        # Create async client
        self.client = AsyncCartesia(api_key=self.api_key)

        # Create WebSocket connection
        self.ws = await self.client.tts.websocket()

        # Start the input queue processor
        self._processor_task = asyncio.create_task(self._process_input_queue())

    async def send(self, input: Input):
        """
        Send input to the Cartesia TTS model.

        Args:
            input: Can be str (text to convert to speech), AudioFrame (ignored for TTS-only),
                   or Image (not supported)
        """
        if isinstance(input, str):
            log_info(f"Cartesia TTS: Queueing text: {input}")
            # Queue the text input for processing
            await self._input_queue.put(input)

        elif isinstance(input, AudioFrame):
            # Cartesia TTS doesn't process audio input, only text
            # This is a TTS-only model, so we ignore audio frames
            log_info("AudioFrame input ignored (Cartesia TTS is text-to-speech only)")

        else:
            log_info(f"Unsupported input type for Cartesia TTS: {type(input)}")

    async def _process_input_queue(self):
        """Background task to process queued text inputs sequentially."""
        log_info("Starting Cartesia input queue processor")

        while not self._closed:
            try:
                # Get next text input from queue (with timeout to check _closed periodically)
                try:
                    text = await asyncio.wait_for(self._input_queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue

                while not self._input_queue.empty():
                    text += await self._input_queue.get()

                speakable, incomplete = parse_speakeable_text(self._pending_text + text)
                speakable, incomplete = aggreagate_sentences(speakable, incomplete)

                self._pending_text = incomplete if incomplete else ""

                if speakable:
                    await self._receive_and_convert(speakable)

            except Exception as e:
                log_info(f"Error in input queue processor: {e}")
                # Continue processing other inputs even if one fails

        log_info("Cartesia input queue processor stopped")

    async def _receive_and_convert(self, text: str):
        """Background task to receive audio from Cartesia and convert to AudioFrames."""
        try:
            self._receiving = True

            ctx = self.ws.context(self._context_id)

            await ctx.send(
                model_id=self.model_id,
                transcript=text,
                voice={"mode": "id", "id": self.voice_id},
                continue_=False,
                stream=True,
                output_format={
                    "container": "raw",
                    "encoding": "pcm_s16le",
                    "sample_rate": SAMPLE_RATE,
                },
                add_timestamps=False,
                add_phoneme_timestamps=False,
                use_original_timestamps=False,
                # Latest SDK only: pronunciation_dict_id=None,
            )

            output_generator = ctx.receive()

            # # Send the text and stream the audio response
            # output_generator = await self.ws.send(
            #     model_id=self.model_id,
            #     transcript=text,
            #     voice={"mode": "id", "id": self.voice_id},
            #     context_id=self._context_id,
            #     stream=True,
            #     output_format={
            #         "container": "raw",
            #         "encoding": "pcm_s16le",
            #         "sample_rate": SAMPLE_RATE,
            #     },
            # )

            async for output in output_generator:
                if not self._receiving:
                    break

                if output.audio:
                    # The format is pcm_s16le (16-bit signed integer little-endian PCM)
                    # which is already in the correct format, so we can use it directly
                    bytes_per_sample = 2  # 16-bit = 2 bytes
                    num_samples = len(output.audio) // bytes_per_sample

                    # log_info(f"Cartesia TTS: Received {num_samples} samples")

                    if num_samples > 0:
                        # Create AudioFrame with s16 format (16-bit signed integer)
                        frame = AudioFrame(format="s16", layout="mono", samples=num_samples)
                        frame.sample_rate = SAMPLE_RATE

                        # Audio data is already in s16le format, just copy it
                        frame.planes[0].update(output.audio)

                        # Put the frame in the output queue
                        await self._output_queue.put(frame)
        except Exception as e:
            log_info(f"Error in receive and convert task: {e}")

        finally:
            self._receiving = False

    async def recv(self) -> AsyncIterator[Output]:
        """
        Receive audio output from Cartesia TTS.

        Yields:
            AudioFrame: Audio frames containing synthesized speech
        """
        while not self._closed:
            try:
                item = await asyncio.wait_for(self._output_queue.get(), timeout=0.1)

                if isinstance(item, AudioFrame):
                    yield item

            except asyncio.TimeoutError:
                # No data available, continue loop
                continue
            except Exception as e:
                log_info(f"Error receiving from Cartesia: {e}")
                break

        log_info("Cartesia TTS: End of stream")

    async def reset(self):
        """Clear queued messages."""
        self._receiving = False
        self._pending_text = ""

        while not self._input_queue.empty():
            self._input_queue.get_nowait()

        while not self._output_queue.empty():
            self._output_queue.get_nowait()

    async def close(self):
        """Close the Cartesia WebSocket connection."""
        log_info("Closing Cartesia TTS session")
        self._closed = True

        # Cancel the processor task
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass

        try:
            if self.ws:
                await self.ws.close()
        except Exception as e:
            log_info(f"Error closing Cartesia WebSocket: {e}")


if __name__ == "__main__":
    import asyncio
    import logging

    import numpy as np
    import sounddevice as sd

    # Configure logging to see output
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    async def main():
        print("Starting Cartesia TTS test with audio playback...", flush=True)
        cartesia = CartesiaTTS()
        print("Connecting...", flush=True)
        await cartesia.connect()
        print("Connected! Sending text...", flush=True)
        await cartesia.send(
            '<emotion value="sad" />Hello, how are you? [laughter]This is a test of the Cartesia text-to-speech system.'
        )
        print("Text sent! Playing audio...", flush=True)

        # Initialize audio playback
        stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
        )
        stream.start()

        # Collect and play frames
        frame_count = 0
        max_wait = 30  # Wait up to 30 seconds

        import time

        start_time = time.time()

        try:
            async for frame in cartesia.recv():
                # Convert AudioFrame to numpy array for sounddevice
                # The frame data is already in s16 format
                audio_data = bytes(frame.planes[0])
                audio_array = np.frombuffer(audio_data, dtype=np.int16)

                # Play the audio
                stream.write(audio_array)

                frame_count += 1
                if frame_count % 10 == 0:
                    print(f"Played {frame_count} frames...", flush=True)

                if (time.time() - start_time) > max_wait:
                    print("Timeout reached", flush=True)
                    break

        finally:
            # Clean up
            stream.stop()
            stream.close()

        print(f"\nTotal frames played: {frame_count}", flush=True)
        print("Closing...", flush=True)
        await cartesia.close()
        print("Done!", flush=True)

    asyncio.run(main())
