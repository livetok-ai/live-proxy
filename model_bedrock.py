import contextlib
from typing import AsyncGenerator, AsyncIterator
import asyncio
import json
import uuid
from aws_sdk_bedrock_runtime.client import BedrockRuntimeClient, InvokeModelWithBidirectionalStreamOperationInput
from aws_sdk_bedrock_runtime.models import InvokeModelWithBidirectionalStreamInputChunk, BidirectionalInputPayloadPart
from aws_sdk_bedrock_runtime.config import Config, HTTPAuthSchemeResolver, SigV4AuthScheme
from smithy_aws_core.credentials_resolvers.environment import EnvironmentCredentialsResolver

from model import Model, Input, Output

class Bedrock(Model):
    def __init__(self, model_id, region):
        self.model_id = model_id
        self.region = region
        self.client = None
        self.stream = None
        self.prompt_name = str(uuid.uuid4())
        self.content_name = str(uuid.uuid4())
        self._input_queue = asyncio.Queue()
        self._response_queue = asyncio.Queue()
        self._response_task = None
        self.is_active = False

    def _initialize_client(self):
        config = Config(
            endpoint_uri=f"https://bedrock-runtime.{self.region}.amazonaws.com",
            region=self.region,
            aws_credentials_identity_resolver=EnvironmentCredentialsResolver(),
            http_auth_scheme_resolver=HTTPAuthSchemeResolver(),
            http_auth_schemes={"aws.auth#sigv4": SigV4AuthScheme()}
        )
        self.client = BedrockRuntimeClient(config=config)

    async def send(self, input: Input):
        if not isinstance(input, str):
            raise NotImplementedError("Bedrock only supports text input for now.")
        await self._input_queue.put(input)

    async def _start_session(self):
        if not self.client:
            self._initialize_client()
        self.stream = await self.client.invoke_model_with_bidirectional_stream(
            InvokeModelWithBidirectionalStreamOperationInput(model_id=self.model_id)
        )
        self.is_active = True
        # Send session start event
        session_start = json.dumps({
            "event": {
                "sessionStart": {
                    "inferenceConfiguration": {
                        "maxTokens": 1024,
                        "topP": 0.9,
                        "temperature": 0.7
                    }
                }
            }
        })
        await self._send_event(session_start)
        # Send prompt start event
        prompt_start = json.dumps({
            "event": {
                "promptStart": {
                    "promptName": self.prompt_name,
                    "textOutputConfiguration": {"mediaType": "text/plain"}
                }
            }
        })
        await self._send_event(prompt_start)
        # Send content start event
        text_content_start = json.dumps({
            "event": {
                "contentStart": {
                    "promptName": self.prompt_name,
                    "contentName": self.content_name,
                    "type": "TEXT",
                    "interactive": True,
                    "role": "USER",
                    "textInputConfiguration": {"mediaType": "text/plain"}
                }
            }
        })
        await self._send_event(text_content_start)

    async def _send_event(self, event_json):
        event = InvokeModelWithBidirectionalStreamInputChunk(
            value=BidirectionalInputPayloadPart(bytes_=event_json.encode('utf-8'))
        )
        await self.stream.input_stream.send(event)

    async def _response_worker(self):
        try:
            while self.is_active:
                output = await self.stream.await_output()
                result = await output[1].receive()
                if result.value and result.value.bytes_:
                    response_data = result.value.bytes_.decode('utf-8')
                    json_data = json.loads(response_data)
                    if 'event' in json_data and 'textOutput' in json_data['event']:
                        text = json_data['event']['textOutput']['content']
                        await self._response_queue.put(text)
        except Exception as e:
            await self._response_queue.put(f"[Error: {e}]")

    async def recv(self) -> AsyncIterator[Output]:
        if not self.is_active:
            await self._start_session()
            self._response_task = asyncio.create_task(self._response_worker())
        # Send user input as textInput event
        input_text = await self._input_queue.get()
        text_input = json.dumps({
            "event": {
                "textInput": {
                    "promptName": self.prompt_name,
                    "contentName": self.content_name,
                    "content": input_text
                }
            }
        })
        await self._send_event(text_input)
        # End content
        text_content_end = json.dumps({
            "event": {
                "contentEnd": {
                    "promptName": self.prompt_name,
                    "contentName": self.content_name
                }
            }
        })
        await self._send_event(text_content_end)
        # Yield responses as they arrive
        while True:
            text = await self._response_queue.get()
            if text is None:
                break
            yield text
            # For now, yield once per input (single response)
            break

    async def close(self):
        self.is_active = False
        if self._response_task:
            self._response_task.cancel()
        if self.stream:
            await self.stream.input_stream.close()
        self.client = None
        self.stream = None

@contextlib.asynccontextmanager
async def connect_bedrock(model_id: str = "amazon.nova-sonic-v1:0", region: str = "us-east-1") -> AsyncGenerator[Bedrock, None]:
    model = Bedrock(model_id=model_id, region=region)
    try:
        yield model
    finally:
        await model.close()
