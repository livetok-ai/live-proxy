import asyncio
import socket

from aiortc import RTCSessionDescription
from aiortc.codecs.g711 import PcmuDecoder, PcmuEncoder
from aiortc.jitterbuffer import JitterFrame
from aiortc.rtcrtpreceiver import RemoteStreamTrack
from aiortc.rtp import RtpPacket
from pyee.asyncio import AsyncIOEventEmitter


class SimplePeerConnection(AsyncIOEventEmitter):
    def __init__(self, public_ip):
        super().__init__()
        self.__socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.__socket.bind(("0.0.0.0", 0))
        self.__socket.setblocking(False)
        self.__public_ip = public_ip
        self.__remote_address = None
        self.__local_track = None
        self.__remote_track = RemoteStreamTrack(kind="audio")
        self.__pcmu_decoder = PcmuDecoder()
        self.__pcmu_encoder = PcmuEncoder()
        self.__sequence_number = 0
        self.__ssrc = 12345  # Random SSRC identifier

    @property
    def connectionState(self) -> str:
        return "connected" if self.__socket else "closed"

    @property
    def localDescription(self):
        sdp = (
            "v=0\r\n"
            f"o=root 371029117 371029117 IN IP4 {self.__public_ip}\r\n"
            "s=Live Proxy\r\n"
            f"c=IN IP4 {self.__public_ip}\r\n"
            "t=0 0\r\n"
            f"m=audio {self.__socket.getsockname()[1]} RTP/AVP 0\r\n"
            "a=rtpmap:0 PCMU/8000\r\n\r\n"
        )
        return RTCSessionDescription(sdp=sdp, type="answer")

    def addTrack(self, track):
        if self.__local_track:
            return
        self.__local_track = track
        asyncio.create_task(self.run_send())

    async def setRemoteDescription(self, description):
        pass

    async def createAnswer(self):
        return self.localDescription

    async def setLocalDescription(self, description):
        asyncio.create_task(self.run_recv())

    async def run_recv(self):
        loop = asyncio.get_event_loop()
        timestamp = 0
        while self.__socket:
            try:
                data, addr = await loop.sock_recvfrom(self.__socket, 1024)

                if not self.__remote_address:
                    self.__remote_address = addr
                    self.emit("connectionstatechange")
                    self.emit("track", self.__remote_track)

                # Remove RTP header (12 bytes)
                if len(data) < 12:
                    continue
                pcmu_payload = data[12:]

                # Create JitterFrame for the decoder (simple wrapper with data and timestamp)
                jitter_frame = JitterFrame(data=pcmu_payload, timestamp=timestamp)

                # Decode PCMU to PCM using aiortc's PcmuDecoder
                decoded_frames = self.__pcmu_decoder.decode(jitter_frame)

                # Send each decoded frame to the remote track queue
                for audio_frame in decoded_frames:
                    self.__remote_track._queue.put_nowait(audio_frame)

                # Increment timestamp (160 samples per 20ms frame at 8kHz)
                timestamp += 160
            except (OSError, ValueError) as e:
                if not self.__socket:
                    break
                print(f"Error receiving data: {e}")

    async def run_send(self):
        loop = asyncio.get_event_loop()
        while self.__socket and self.__local_track:
            try:
                # Receive AudioFrame from the track
                av_frame = await self.__local_track.recv()

                # Wait until we have a remote address (connection established)
                if not self.__remote_address:
                    continue

                # Encode the audio frame to PCMU format
                payloads, timestamp = self.__pcmu_encoder.encode(av_frame)

                # Send each encoded payload as an RTP packet
                for i, payload in enumerate(payloads):
                    # Set marker bit to 1 for the last packet in the frame
                    marker = 1 if i == len(payloads) - 1 else 0

                    # Create RTP packet
                    rtp_packet = RtpPacket(
                        payload_type=0,  # PCMU payload type
                        marker=marker,
                        sequence_number=self.__sequence_number,
                        timestamp=timestamp,
                        ssrc=self.__ssrc,
                        payload=payload,
                    )

                    # Serialize RTP packet to bytes
                    packet_bytes = rtp_packet.serialize()

                    # Send the RTP packet
                    await loop.sock_sendto(self.__socket, packet_bytes, self.__remote_address)

                    # Increment sequence number for next packet
                    self.__sequence_number = (self.__sequence_number + 1) % 65536

            except (OSError, ValueError) as e:
                if not self.__socket:
                    break
                print(f"Error sending data: {e}")

    async def close(self):
        if self.__socket:
            self.__socket.close()
            self.__socket = None
