import asyncio
from typing import AsyncIterator

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from logger import log_info
from model import Input, Model, Output


class TextSentimentProvider(Model):
    @property
    def supports_text(self) -> bool:
        return True

    @classmethod
    async def setup(cls):

        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError:
            log_info("Downloading NLTK VADER lexicon...")
            # Download in an executor to avoid blocking the main event loop
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: nltk.download("vader_lexicon", quiet=True))

    def __init__(self, name=None, connection=None, **kwargs):
        super().__init__(name=name, connection=connection, **kwargs)
        self.sia = None
        self.last_sentiment = None
        self.model = kwargs.get("model") or name
        log_info(f"Text Sentiment provider model: {self.model}")

    async def connect(self):

        # Download VADER lexicon if not already available
        await TextSentimentProvider.setup()

        self.sia = SentimentIntensityAnalyzer()

    async def handle_transcription(self, text: str):
        if not self.sia or not text:
            return

        # Perform sentiment analysis in a thread pool to be safe
        loop = asyncio.get_event_loop()
        scores = await loop.run_in_executor(None, self.sia.polarity_scores, text)

        compound = scores.get("compound", 0.0)

        # Map VADER compound score to sentiments
        if compound >= 0.05:
            sentiment = "positive"
        elif compound <= -0.05:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        if sentiment != self.last_sentiment:
            log_info(f"Text sentiment changed: {sentiment} (score: {compound:.4f}, text: '{text}')")
            self.last_sentiment = sentiment
            self._emit("sentiment_changed", sentiment)

    async def send(self, input: Input):
        if isinstance(input, str):
            await self.handle_transcription(input)

    async def recv(self) -> AsyncIterator[Output]:
        if False:
            yield

    async def close(self):
        log_info("Closing Text Sentiment provider")
        self.sia = None
