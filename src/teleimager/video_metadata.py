"""Source timestamps associated with the exact RTP frame sent to native viewers."""
import asyncio
import json
import struct
import time


CHANNEL_LABEL = "r1-video-meta-v1"
MAX_BUFFERED_BYTES = 16_384


class VideoMetadataSender:
    def __init__(self, sender, metadata_for_pts, clock_id):
        self._clock_id = clock_id
        self._channel = None
        self._rtp_task = None
        self._pending = None
        original_next = sender._next_encoded_frame
        transport = sender.transport
        original_send = transport._send_rtp

        async def next_encoded_frame(codec):
            self._pending = None
            encoded = await original_next(codec)
            if encoded is not None:
                metadata = metadata_for_pts(encoded.timestamp)
                if metadata is not None:
                    self._rtp_task = asyncio.current_task()
                    self._pending = (codec.payloadType, metadata)
            return encoded

        async def send_rtp(data):
            # Retransmission and RTCP run in separate tasks. They must never
            # associate an old packet with the frame currently being encoded.
            pending = self._pending
            is_frame_end = (
                pending is not None
                and asyncio.current_task() is self._rtp_task
                and len(data) >= 12
                and data[0] >> 6 == 2
                and data[1] & 0x80
                and data[1] & 0x7F == pending[0]
                and struct.unpack_from("!I", data, 8)[0] == sender._ssrc
            )
            await original_send(data)
            if is_frame_end:
                self._pending = None
                self._send({
                    "type": "frame", "v": 1,
                    "rtp_timestamp": struct.unpack_from("!I", data, 4)[0],
                    **pending[1],
                    "send_monotonic_ns": time.monotonic_ns(),
                })

        sender._next_encoded_frame = next_encoded_frame
        transport._send_rtp = send_rtp

    def attach(self, channel):
        if (channel.label != CHANNEL_LABEL or channel.ordered
                or channel.maxRetransmits != 0 or channel.maxPacketLifeTime is not None):
            return
        self._channel = channel

        @channel.on("message")
        def message(data):
            received_ns = time.monotonic_ns()
            if not isinstance(data, str) or len(data) > 2048:
                return
            try:
                request = json.loads(data)
            except (ValueError, UnicodeDecodeError):
                return
            if (not isinstance(request, dict) or request.get("type") != "ping"
                    or request.get("v") != 1
                    or type(request.get("id")) is not int
                    or not 0 <= request["id"] <= 2**53 - 1
                    or type(request.get("t0_ns")) is not int
                    or not 0 < request["t0_ns"] <= 2**64 - 1):
                return
            self._send({
                "type": "pong", "v": 1, "id": request["id"], "t0_ns": request["t0_ns"],
                "clock_id": self._clock_id, "receive_monotonic_ns": received_ns,
                "send_monotonic_ns": time.monotonic_ns(),
            })

    def _send(self, payload):
        channel = self._channel
        if channel is None or channel.readyState != "open":
            return
        message = json.dumps(payload, separators=(",", ":"), allow_nan=False)
        if channel.bufferedAmount + len(message) <= MAX_BUFFERED_BYTES:
            channel.send(message)
