from unittest.mock import patch

import pytest

from providers.text_sentiment.text_sentiment import TextSentimentProvider


@pytest.mark.asyncio
async def test_text_sentiment_provider_init():
    """Test TextSentimentProvider initialization."""
    provider = TextSentimentProvider()
    assert provider.sia is None
    assert provider.last_sentiment is None


@pytest.mark.asyncio
async def test_text_sentiment_provider_connect():
    """Test TextSentimentProvider connect loading VADER lexicon."""
    provider = TextSentimentProvider()

    with patch("nltk.download"):
        await provider.connect(
            model="text_sentiment",
            system_instructions=None,
            tools=None,
            tool_callback=None,
            voice=None,
            language=None,
            api_key=None,
        )
        assert provider.sia is not None

    await provider.close()


@pytest.mark.asyncio
async def test_text_sentiment_provider_handle_transcription():
    """Test TextSentimentProvider sentiment polarity analysis."""
    provider = TextSentimentProvider()
    await provider.connect(model="text_sentiment")

    # Analyze positive sentiment
    with patch.object(provider, "_emit") as mock_emit:
        await provider.handle_transcription("This is an absolutely amazing and wonderful day!")
        assert provider.last_sentiment == "positive"
        mock_emit.assert_called_once_with("sentiment_changed", "positive")

    # Analyze negative sentiment (should trigger change and emit)
    with patch.object(provider, "_emit") as mock_emit:
        await provider.handle_transcription("This is terrible, I hate this so much.")
        assert provider.last_sentiment == "negative"
        mock_emit.assert_called_once_with("sentiment_changed", "negative")

    # Analyze neutral sentiment (should trigger change and emit)
    with patch.object(provider, "_emit") as mock_emit:
        await provider.handle_transcription("The table is green.")
        assert provider.last_sentiment == "neutral"
        mock_emit.assert_called_once_with("sentiment_changed", "neutral")

    await provider.close()


@pytest.mark.asyncio
async def test_text_sentiment_provider_send():
    """Test TextSentimentProvider handles text input via send."""
    provider = TextSentimentProvider()
    await provider.connect(model="text_sentiment")

    with patch.object(provider, "handle_transcription") as mock_handle:
        await provider.send("Hello there")
        mock_handle.assert_called_once_with("Hello there")

    await provider.close()
