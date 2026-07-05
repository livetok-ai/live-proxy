"""Tests for the WebTransport/raw-QUIC transport."""

import asyncio
import json
import socket
import ssl

import pytest
from aioquic.asyncio import QuicConnectionProtocol
from aioquic.asyncio import connect as quic_connect
from aioquic.h3.connection import H3_ALPN, H3Connection
from aioquic.h3.events import HeadersReceived
from aioquic.quic.configuration import QuicConfiguration

from webtransport.protocol import (
    FRAME_TYPE_AUDIO,
    FRAME_TYPE_CONTROL,
    FRAME_TYPE_VIDEO,
    FrameDecoder,
    pack_audio,
    pack_control,
    pack_video,
)
from webtransport.server import RAW_QUIC_ALPN, WebTransportServer
from webtransport.session import WebTransportPeerConnection


class TestFraming:
    def test_roundtrip_audio(self):
        decoder = FrameDecoder()
        payload = b"\x01\x02\x03\x04"
        frames = decoder.feed(pack_audio(payload, sample_rate=16000, channels=1, timestamp_us=42))

        assert len(frames) == 1
        frame = frames[0]
        assert frame.type == FRAME_TYPE_AUDIO
        assert frame.timestamp_us == 42
        assert frame.extra == {"sample_rate": 16000, "channels": 1}
        assert frame.payload == payload

    def test_roundtrip_video(self):
        decoder = FrameDecoder()
        payload = b"fakejpegbytes"
        frames = decoder.feed(pack_video(payload, timestamp_us=7, keyframe=True))

        assert frames[0].type == FRAME_TYPE_VIDEO
        assert frames[0].extra == {"keyframe": True}
        assert frames[0].payload == payload

    def test_roundtrip_control(self):
        decoder = FrameDecoder()
        payload = json.dumps({"model": "yolo"}).encode("utf-8")
        frames = decoder.feed(pack_control(payload))

        assert frames[0].type == FRAME_TYPE_CONTROL
        assert json.loads(frames[0].payload) == {"model": "yolo"}

    def test_partial_and_multiple_frames_split_across_chunks(self):
        decoder = FrameDecoder()
        first = pack_control(b"a")
        data = first + pack_control(b"bb")

        frames = decoder.feed(data[:3])
        assert frames == []

        # Deliver the rest of the first frame plus the whole second frame in one chunk.
        frames = decoder.feed(data[3:])
        assert len(frames) == 2
        assert frames[0].payload == b"a"
        assert frames[1].payload == b"bb"


class TestWebTransportPeerConnection:
    def test_control_frame_emits_datachannel_then_message(self):
        sent = []
        pc = WebTransportPeerConnection(send_bytes=sent.append)

        channels = []
        pc.on("datachannel", lambda ch: channels.append(ch))

        decoder = FrameDecoder()
        frame = decoder.feed(pack_control(b'{"hello": 1}'))[0]
        pc.handle_frame(frame)

        assert len(channels) == 1
        messages = []
        channels[0].on("message", lambda m: messages.append(m))

        frame2 = decoder.feed(pack_control(b'{"hello": 2}'))[0]
        pc.handle_frame(frame2)
        assert messages == ['{"hello": 2}']

        channels[0].send('{"reply": true}')
        assert len(sent) == 1

    @pytest.mark.asyncio
    async def test_audio_frame_emits_track_with_decoded_pcm(self):
        pc = WebTransportPeerConnection(send_bytes=lambda b: None)
        tracks = []
        pc.on("track", lambda t: tracks.append(t))

        pcm = b"\x10\x00" * 160  # 160 s16le samples
        decoder = FrameDecoder()
        frame = decoder.feed(pack_audio(pcm, sample_rate=16000, channels=1))[0]
        pc.handle_frame(frame)

        assert len(tracks) == 1
        assert tracks[0].kind == "audio"
        av_frame = await asyncio.wait_for(tracks[0].recv(), timeout=1)
        assert av_frame.sample_rate == 16000
        assert av_frame.samples == 160

    @pytest.mark.asyncio
    async def test_video_frame_emits_track_with_decoded_image(self):
        import io

        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (32, 24), color="red").save(buf, format="JPEG")

        pc = WebTransportPeerConnection(send_bytes=lambda b: None)
        tracks = []
        pc.on("track", lambda t: tracks.append(t))

        decoder = FrameDecoder()
        frame = decoder.feed(pack_video(buf.getvalue()))[0]
        pc.handle_frame(frame)

        assert tracks[0].kind == "video"
        av_frame = await asyncio.wait_for(tracks[0].recv(), timeout=1)
        image = av_frame.to_image()
        assert image.size == (32, 24)


def _free_udp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class TestRawQuicIntegration:
    """Drives the real aioquic client against our server (raw QUIC ALPN, no HTTP/3)."""

    @pytest.mark.asyncio
    async def test_session_bootstrap_and_audio_frame(self):
        port = _free_udp_port()
        sessions = []

        async def on_session(pc, params):
            sessions.append((pc, params))

        server = WebTransportServer(host="127.0.0.1", port=port, on_session=on_session)
        await server.bind()
        try:
            client_config = QuicConfiguration(is_client=True, alpn_protocols=[RAW_QUIC_ALPN])
            client_config.verify_mode = ssl.CERT_NONE  # trust the ephemeral self-signed cert for this test

            async with quic_connect("127.0.0.1", port, configuration=client_config) as protocol:
                reader, writer = await protocol.create_stream()

                writer.write(pack_control(json.dumps({"model": "", "metadata": {"test": True}}).encode("utf-8")))
                writer.write(pack_audio(b"\x00\x01" * 160, sample_rate=16000, channels=1))
                await asyncio.sleep(0.2)

                assert len(sessions) == 1
                pc, params = sessions[0]
                assert params == {"model": "", "metadata": {"test": True}}
                assert pc.connectionState == "connected"

                await pc.close()
        finally:
            server.close()


class _H3ClientProtocol(QuicConnectionProtocol):
    """Minimal client-side H3/WebTransport driver, standing in for a browser's WebTransport API."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.http = H3Connection(self._quic, enable_webtransport=True)
        self.events: asyncio.Queue = asyncio.Queue()

    def quic_event_received(self, event) -> None:
        for h3_event in self.http.handle_event(event):
            self.events.put_nowait(h3_event)

    async def open_webtransport_session(self, authority: str, path: str) -> int:
        session_id = self._quic.get_next_available_stream_id()
        self.http.send_headers(
            session_id,
            [
                (b":method", b"CONNECT"),
                (b":scheme", b"https"),
                (b":authority", authority.encode()),
                (b":path", path.encode()),
                (b":protocol", b"webtransport"),
            ],
        )
        self.transmit()

        while True:
            event = await asyncio.wait_for(self.events.get(), timeout=5)
            if isinstance(event, HeadersReceived) and event.stream_id == session_id:
                status = dict(event.headers).get(b":status")
                assert status == b"200", f"CONNECT rejected: {status}"
                return session_id

    def send_webtransport_data(self, session_id: int, data: bytes) -> int:
        stream_id = self.http.create_webtransport_stream(session_id)
        self._quic.send_stream_data(stream_id, data)
        self.transmit()
        return stream_id


class TestWebTransportH3Integration:
    """Drives a real HTTP/3 CONNECT + WebTransport stream against our server."""

    @pytest.mark.asyncio
    async def test_session_bootstrap_over_webtransport(self):
        port = _free_udp_port()
        sessions = []

        async def on_session(pc, params):
            sessions.append((pc, params))

        server = WebTransportServer(host="127.0.0.1", port=port, on_session=on_session)
        await server.bind()
        try:
            client_config = QuicConfiguration(is_client=True, alpn_protocols=H3_ALPN, max_datagram_frame_size=65536)
            client_config.verify_mode = ssl.CERT_NONE

            async with quic_connect(
                "127.0.0.1", port, configuration=client_config, create_protocol=_H3ClientProtocol
            ) as protocol:
                session_id = await protocol.open_webtransport_session("127.0.0.1", "/webtransport")

                init = pack_control(json.dumps({"model": "", "metadata": {"via": "h3"}}).encode("utf-8"))
                stream_id = protocol.send_webtransport_data(session_id, init)
                protocol._quic.send_stream_data(stream_id, pack_audio(b"\x00\x01" * 160, sample_rate=16000, channels=1))
                protocol.transmit()

                await asyncio.sleep(0.2)

                assert len(sessions) == 1
                pc, params = sessions[0]
                assert params == {"model": "", "metadata": {"via": "h3"}}
                assert pc.connectionState == "connected"

                await pc.close()
        finally:
            server.close()
