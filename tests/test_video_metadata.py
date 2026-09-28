import asyncio
import importlib.util
import json
from pathlib import Path
import struct
from types import SimpleNamespace
import unittest

MODULE = Path(__file__).resolve().parents[1] / "src/teleimager/video_metadata.py"
spec = importlib.util.spec_from_file_location("video_metadata", MODULE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Channel:
    label = module.CHANNEL_LABEL
    ordered = False
    maxRetransmits = 0
    maxPacketLifeTime = None
    readyState = "open"
    bufferedAmount = 0

    def __init__(self):
        self.messages = []
        self.handlers = {}

    def on(self, event):
        def register(callback):
            self.handlers[event] = callback
        return register

    def send(self, payload):
        self.messages.append(json.loads(payload))


class Sender:
    _ssrc = 123456

    def __init__(self):
        self.transport = SimpleNamespace(_send_rtp=self.send)
        self.frames = []
        self.packets = []
        self.fail_send = False

    async def _next_encoded_frame(self, codec):
        return self.frames.pop(0)

    async def send(self, data):
        if self.fail_send:
            raise ConnectionError("closed")
        self.packets.append(data)


def packet(timestamp, marker=True, payload_type=101, ssrc=123456):
    return struct.pack("!BBHII", 0x80, payload_type | (0x80 if marker else 0), 2, timestamp, ssrc) + b"payload"


class MetadataTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.sender = Sender()
        self.source = {
            42: {"source_sequence": 9, "source_monotonic_ns": 123456789,
                 "clock_id": "boot", "source_epoch": "epoch", "timestamp_kind": "pc2_rtp_decoded_receive"},
            43: {"source_sequence": 10, "source_monotonic_ns": 123556789,
                 "clock_id": "boot", "source_epoch": "epoch", "timestamp_kind": "pc2_rtp_decoded_receive"},
        }
        self.adapter = module.VideoMetadataSender(self.sender, self.source.get, "boot")
        self.channel = Channel()
        self.adapter.attach(self.channel)
        self.codec = SimpleNamespace(payloadType=101)

    async def encode(self, pts=42):
        self.sender.frames.append(SimpleNamespace(timestamp=pts))
        return await self.sender._next_encoded_frame(self.codec)

    async def test_actual_timestamp_wrap_and_last_packet_only(self):
        await self.encode()
        await self.sender.transport._send_rtp(packet(0xFFFFFFFE, marker=False))
        self.assertFalse(self.channel.messages)
        await self.sender.transport._send_rtp(packet(0xFFFFFFFE))
        frame = self.channel.messages[0]
        self.assertEqual(frame["rtp_timestamp"], 0xFFFFFFFE)
        self.assertEqual(frame["source_sequence"], 9)
        self.assertEqual(frame["source_monotonic_ns"], 123456789)
        self.assertEqual(frame["source_epoch"], "epoch")
        self.assertEqual(frame["v"], 1)
        await self.encode(43)
        await self.sender.transport._send_rtp(packet(10))
        self.assertEqual(self.channel.messages[-1]["rtp_timestamp"], 10)
        self.assertEqual(self.channel.messages[-1]["source_sequence"], 10)

    async def test_retransmission_rtcp_and_other_stream_cannot_consume_metadata(self):
        await self.encode()
        await asyncio.create_task(self.sender.transport._send_rtp(packet(99)))
        await self.sender.transport._send_rtp(packet(88, payload_type=102))
        await self.sender.transport._send_rtp(packet(77, ssrc=1))
        await self.sender.transport._send_rtp(packet(66, payload_type=72))
        self.assertFalse(self.channel.messages)
        await self.sender.transport._send_rtp(packet(55))
        self.assertEqual(self.channel.messages[0]["rtp_timestamp"], 55)
        await self.sender.transport._send_rtp(packet(55))
        self.assertEqual(len(self.channel.messages), 1)

    async def test_missing_source_metadata_never_uses_previous_frame(self):
        await self.encode()
        await self.sender.transport._send_rtp(packet(100))
        await self.encode(999)
        await self.sender.transport._send_rtp(packet(200))
        self.assertEqual(len(self.channel.messages), 1)

    async def test_dropped_encoder_output_clears_pending(self):
        await self.encode()
        self.sender.frames.append(None)
        self.assertIsNone(await self.sender._next_encoded_frame(self.codec))
        await self.sender.transport._send_rtp(packet(100))
        self.assertFalse(self.channel.messages)

    async def test_failed_send_does_not_report_frame(self):
        await self.encode()
        self.sender.fail_send = True
        with self.assertRaises(ConnectionError):
            await self.sender.transport._send_rtp(packet(100))
        self.assertFalse(self.channel.messages)

    async def test_closed_or_backlogged_channel_drops_metadata(self):
        self.channel.bufferedAmount = module.MAX_BUFFERED_BYTES
        await self.encode()
        await self.sender.transport._send_rtp(packet(100))
        self.assertFalse(self.channel.messages)
        self.channel.bufferedAmount = 0
        self.channel.readyState = "closed"
        await self.encode(43)
        await self.sender.transport._send_rtp(packet(200))
        self.assertFalse(self.channel.messages)
        self.channel.readyState = "open"
        self.assertFalse(self.channel.messages)

    async def test_ping_echo_and_monotonic_times(self):
        self.channel.handlers["message"](json.dumps({"type": "ping", "v": 1, "id": 7, "t0_ns": 123456}))
        reply = self.channel.messages[0]
        self.assertEqual((reply["type"], reply["id"], reply["t0_ns"], reply["clock_id"]), ("pong", 7, 123456, "boot"))
        self.assertLessEqual(reply["receive_monotonic_ns"], reply["send_monotonic_ns"])

    async def test_malformed_pings_and_reliable_channel_are_rejected(self):
        for value in ["bad", "[]", "{}", b"data", "x" * 2049,
                      json.dumps({"type": "ping", "v": 1, "id": True, "t0_ns": 1}),
                      json.dumps({"type": "ping", "v": 1, "id": 1, "t0_ns": -1})]:
            self.channel.handlers["message"](value)
        self.assertFalse(self.channel.messages)
        other = Channel()
        other.ordered = True
        self.adapter.attach(other)
        self.assertFalse(other.handlers)


if __name__ == "__main__":
    unittest.main(verbosity=2)
