"""Support for the webrtc.org "playout-delay" RTP header extension.

aiortc (1.14.0) has no built-in notion of this extension: `aiortc.rtp.HeaderExtensions`
is a fixed dataclass and `aiortc.codecs.HEADER_EXTENSIONS` only lists mid/abs-send-time/
audio-level. This module monkeypatches both so the extension is:

  1. Offered/answered in SDP as ``a=extmap:<id> http://www.webrtc.org/experiments/rtp-hdrext/playout-delay``
     on video m-lines.
  2. Encoded with a constant min/max delay on every outgoing video RTP packet, once the
     remote peer has negotiated it (aiortc keeps whatever extmap id the remote assigned,
     see `RTCPeerConnection.find_common_header_extensions`).

Spec: https://webrtc.googlesource.com/src/+/refs/heads/main/docs/native-code/rtp-hdrext/playout-delay
3-byte value: 12-bit MinDelay | 12-bit MaxDelay, in units of 10ms (range 0-40950ms each).
"""

from struct import pack

from aiortc import rtp as aiortc_rtp
from aiortc.codecs import HEADER_EXTENSIONS
from aiortc.rtcrtpparameters import RTCRtpHeaderExtensionParameters

PLAYOUT_DELAY_URI = "http://www.webrtc.org/experiments/rtp-hdrext/playout-delay"
PLAYOUT_DELAY_EXTENSION_ID = 5

_installed = False


def _pack_playout_delay(min_ms: int, max_ms: int) -> bytes:
    min_units = min(max(min_ms // 10, 0), 0xFFF)
    max_units = min(max(max_ms // 10, 0), 0xFFF)
    value = (min_units << 12) | max_units
    return pack("!L", value)[1:]


def install_playout_delay_extension(min_ms: int = 0, max_ms: int = 0) -> None:
    """Advertise the playout-delay extension and send a constant value on every packet.

    Safe to call more than once; only the first call takes effect.
    """
    global _installed
    if _installed:
        return
    _installed = True

    HEADER_EXTENSIONS["video"].append(
        RTCRtpHeaderExtensionParameters(id=PLAYOUT_DELAY_EXTENSION_ID, uri=PLAYOUT_DELAY_URI)
    )

    packed_value = _pack_playout_delay(min_ms, max_ms)

    original_configure = aiortc_rtp.HeaderExtensionsMap.configure
    original_set = aiortc_rtp.HeaderExtensionsMap.set

    def configure(self, parameters):
        original_configure(self, parameters)
        self._playout_delay_id = None
        for ext in parameters.headerExtensions:
            if ext.uri == PLAYOUT_DELAY_URI:
                self._playout_delay_id = ext.id

    def set(self, values):
        profile, value = original_set(self, values)
        playout_delay_id = getattr(self, "_playout_delay_id", None)
        if playout_delay_id is None:
            return profile, value
        extensions = aiortc_rtp.unpack_header_extensions(profile, value)
        extensions.append((playout_delay_id, packed_value))
        return aiortc_rtp.pack_header_extensions(extensions)

    aiortc_rtp.HeaderExtensionsMap.configure = configure
    aiortc_rtp.HeaderExtensionsMap.set = set
