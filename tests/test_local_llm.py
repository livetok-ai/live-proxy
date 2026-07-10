import asyncio
from unittest.mock import MagicMock, patch

import pytest

from model import ModelEvents
from providers.local_llm import LocalLLMProvider


@pytest.mark.asyncio
async def test_local_llm_provider_init():
    """Test LocalLLM provider capability properties and initialization defaults."""
    provider = LocalLLMProvider(name="local_llm")

    assert provider.supports_audio is False
    assert provider.supports_text is True
    assert provider.supports_video is False
    assert provider.video_support is False
    assert provider.model_id == "HuggingFaceTB/SmolLM2-135M-Instruct"
    assert provider.tokenizer is None
    assert provider.model is None


@pytest.mark.asyncio
async def test_local_llm_provider_connect():
    """Test LocalLLM provider connect loads tokenizer and model."""
    provider = LocalLLMProvider(name="local_llm")

    mock_tokenizer = MagicMock()
    mock_model = MagicMock()

    with patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer) as mock_tok_load, patch(
        "transformers.AutoModelForCausalLM.from_pretrained", return_value=mock_model
    ) as mock_model_load:

        await provider.connect()

        assert provider.tokenizer == mock_tokenizer
        assert provider.model == mock_model
        mock_tok_load.assert_called_once_with("HuggingFaceTB/SmolLM2-135M-Instruct")
        mock_model_load.assert_called_once_with("HuggingFaceTB/SmolLM2-135M-Instruct")

    await provider.close()


@pytest.mark.asyncio
async def test_local_llm_provider_generation():
    """Test LocalLLM generation logic, history tracking, and token emission."""
    provider = LocalLLMProvider(name="local_llm")

    mock_tokenizer = MagicMock()
    mock_model = MagicMock()

    # Mock chat template and inputs
    mock_tokenizer.apply_chat_template.return_value = "<prompt>"
    mock_tokenizer.return_value = {"input_ids": [1, 2, 3]}

    provider.tokenizer = mock_tokenizer
    provider.model = mock_model

    emitted_events = []

    def mock_emit(event_type, data):
        emitted_events.append((event_type, data))

    provider._emit = mock_emit

    # Mock TextIteratorStreamer to return tokens and then StopIteration
    class MockStreamer:
        def __init__(self, *args, **kwargs):
            self.tokens = ["Hello", " ", "World", "!"]
            self.idx = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self.idx < len(self.tokens):
                t = self.tokens[self.idx]
                self.idx += 1
                return t
            raise StopIteration

    with patch("providers.local_llm.local_llm.TextIteratorStreamer", MockStreamer), patch(
        "providers.local_llm.local_llm.Thread"
    ):

        # Trigger send
        await provider.send("Hi there")

        # Wait a short duration for the async generation task to finish
        await asyncio.sleep(0.1)

        # Verify that the final response was emitted
        assert ("response", {"text": "Hello World!"}) in emitted_events

    await provider.close()
