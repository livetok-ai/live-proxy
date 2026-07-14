import asyncio
from threading import Thread
from typing import AsyncIterator

from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from logger import log_info
from model import Input, Model, ModelEvents, Output


class LocalLLMProvider(Model):
    @property
    def is_llm(self) -> bool:
        return True

    @property
    def supports_audio(self) -> bool:
        return False

    @property
    def supports_text(self) -> bool:
        return True

    @property
    def supports_video(self) -> bool:
        return False

    @classmethod
    async def setup(cls, model_id: str = "HuggingFaceTB/SmolLM2-135M-Instruct"):
        """Pre-download the model and tokenizer to cache."""
        log_info(f"Setting up and pre-downloading LocalLLM model: {model_id}...")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: AutoTokenizer.from_pretrained(model_id))
        await loop.run_in_executor(None, lambda: AutoModelForCausalLM.from_pretrained(model_id))
        log_info("LocalLLM setup complete.")

    def __init__(self, name=None, connection=None, **kwargs):
        super().__init__(name=name, connection=connection, **kwargs)
        self.model_id = kwargs.get("model") or "HuggingFaceTB/SmolLM2-135M-Instruct"
        self.system_instructions = kwargs.get("system_instructions", None)
        self.tokenizer = None
        self.model = None
        self.generation_cancelled = False
        log_info(f"LocalLLM provider initialized with model: {self.model_id}")

    async def connect(self):
        log_info(f"Connecting LocalLLM model: {self.model_id}...")
        loop = asyncio.get_event_loop()

        # Load model and tokenizer in a thread pool to avoid blocking the main event loop
        self.tokenizer = await loop.run_in_executor(None, lambda: AutoTokenizer.from_pretrained(self.model_id))
        self.model = await loop.run_in_executor(None, lambda: AutoModelForCausalLM.from_pretrained(self.model_id))
        log_info("LocalLLM model loaded and connected successfully.")

    async def send(self, input: Input):
        if not isinstance(input, str):
            return

        self.generation_cancelled = False

        # Construct single-turn messages list (no history accumulation)
        messages = []
        if self.system_instructions:
            messages.append({"role": "system", "content": self.system_instructions})
        messages.append({"role": "user", "content": input})

        await self._generate(messages)

    async def _generate(self, messages):
        log_info(f"LocalLLM generate with input: {messages}")
        try:
            if not self.model or not self.tokenizer:
                log_info("LocalLLM model not connected.")
                return

            # Apply chat template
            input_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(input_text, return_tensors="pt")

            # Setup streamer
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

            # Generate in background thread
            generation_kwargs = dict(
                inputs,
                streamer=streamer,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
            )

            # Run in a separate thread to prevent blocking and log thread exceptions
            def run_generation():
                try:
                    self.model.generate(**generation_kwargs)
                except Exception as ex:
                    log_info(f"Exception in model.generate thread: {ex}")

            thread = Thread(target=run_generation)
            thread.start()

            # Read from streamer asynchronously
            loop = asyncio.get_event_loop()

            full_response = ""
            while True:
                # Get next token/chunk from the streamer (blocks, so run in executor)
                token = await loop.run_in_executor(None, lambda: next(streamer, None))
                if token is None or self.generation_cancelled:
                    break

                # Emit token/chunk
                full_response += token

                log_info("LocalLLM Response  " + full_response)

                # Yield control to the event loop
                await asyncio.sleep(0.01)

            if full_response and not self.generation_cancelled:
                self._emit("response", {"text": full_response})

        except Exception as e:
            log_info(f"Error in LocalLLM generation: {e}")
        log_info("LocalLLM ends")

    async def recv(self) -> AsyncIterator[Output]:
        # LocalLLM is text-only and uses direct event emission,
        # so recv is an empty async generator.
        if False:
            yield

    async def clear(self):
        self.generation_cancelled = True

    async def close(self):
        log_info("Closing LocalLLM provider")
        await self.clear()
        self.model = None
        self.tokenizer = None


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    async def main():
        log_info("Starting LocalLLM standalone test...")
        provider = LocalLLMProvider()

        response_received = False

        def handle_response(data):
            nonlocal response_received
            print(f"EVENT: response -> {data}")
            response_received = True

        # Register simple handlers to inspect what's being emitted
        provider.on(ModelEvents.INPUT_TRANSCRIPTION, lambda data: print(f"EVENT: input_transcription -> {data}"))
        provider.on(ModelEvents.OUTPUT_TRANSCRIPTION, lambda data: print(f"EVENT: output_transcription -> {data}"))
        provider.on("response", handle_response)

        await provider.connect()

        # Send test input
        prompt = "Hello! Who are you?"
        print(f"Sending prompt: {prompt}")
        await provider.send(prompt)

        # Let it generate. SmolLM2 is very fast, but let's wait up to 10 seconds.
        for _ in range(100):
            await asyncio.sleep(0.1)
            if response_received:
                break

        await provider.close()

    asyncio.run(main())
