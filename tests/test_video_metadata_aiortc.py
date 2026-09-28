"""Run in an isolated directory with PC2's existing aiortc environment."""
import asyncio
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import av
import numpy as np
import teleimager
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRelay
from aiortc.rtcrtpsender import RTCRtpSender
from teleimager.timing import jpeg_timestamp

ROOT = Path(__file__).resolve().parents[1]
teleimager.__path__.insert(0, str(ROOT / "src/teleimager"))
spec = importlib.util.spec_from_file_location("teleimager.server_under_test", ROOT / "src/teleimager/server.py")
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)


class SourceTests(unittest.IsolatedAsyncioTestCase):
    def camera(self):
        with patch.object(server, "StereoCapture", return_value=SimpleNamespace(frames=SimpleNamespace(take=lambda **kwargs: self.pair))):
            return server.GstStereoRtpCamera("head", 5002, 5003, {
                "image_shape": [16, 32], "fps": 30,
                "enable_zmq": True, "zmq_port": 55555,
                "enable_webrtc": True, "webrtc_port": 60001,
            })

    def set_pair(self, color):
        stamp = time.monotonic()
        data = np.full((16, 16, 3), color, dtype=np.uint8).tobytes()
        self.pair = [(stamp, 16, 16, 48, data), (stamp + 0.001, 16, 16, 48, data)]

    async def test_source_frame_and_metadata_are_one_ring_sample(self):
        camera = self.camera()
        self.set_pair(20)
        camera._update_frame()
        first = camera.get_webrtc_sample()
        first_jpeg_meta = jpeg_timestamp(camera.get_jpeg_bytes())
        self.set_pair(90)
        camera._update_frame()
        second = camera.get_webrtc_sample()
        self.assertEqual(int(first[0][0, 0, 0]), 20)
        self.assertEqual(int(second[0][0, 0, 0]), 90)
        self.assertEqual(first[1]["source_sequence"], 1)
        self.assertEqual(second[1]["source_sequence"], 2)
        self.assertEqual(first[1]["source_epoch"], second[1]["source_epoch"])
        for key, value in first[1].items():
            self.assertEqual(first_jpeg_meta[key], value)
        self.assertEqual(camera.get_bgr_frame().shape, (16, 32, 3))
        second[1]["source_monotonic_ns"] = time.monotonic_ns() - int(2 * server.MAX_FRAME_AGE * 1e9)
        self.assertIsNone(camera.get_webrtc_sample())
        self.assertNotEqual(camera._source_epoch, self.camera()._source_epoch)

    async def test_bounded_pts_map_and_plain_safari_frames(self):
        track = server.BGRArrayVideoStreamTrack()
        loop = asyncio.get_running_loop()
        with patch.object(server, "time", SimpleNamespace(monotonic=lambda: 1)):
            for index in range(140):
                track.push_frame((np.zeros((16, 32, 3), np.uint8), {"source_sequence": index}), loop)
                await asyncio.sleep(0)
        frame = await track.recv()
        self.assertEqual(frame.pts, 139)
        self.assertEqual(track.metadata_for_pts(frame.pts)["source_sequence"], 139)
        self.assertEqual(len(track._metadata_by_pts), 128)
        self.assertIsNone(track.metadata_for_pts(0))
        track.push_frame(np.zeros((16, 32, 3), np.uint8), loop)
        await asyncio.sleep(0)
        safari_frame = await track.recv()
        self.assertIsInstance(safari_frame, av.VideoFrame)
        self.assertIsNone(track.metadata_for_pts(safari_frame.pts))
        track.stop()

    async def test_real_h264_peer_datachannel_and_exact_frame_association(self):
        publisher = server.WebRTC_PublisherThread(0)
        publisher._bgr_track = server.BGRArrayVideoStreamTrack()
        publisher._relay = MediaRelay()
        client = RTCPeerConnection()
        safari = RTCPeerConnection()
        channel = client.createDataChannel("r1-video-meta-v1", ordered=False, maxRetransmits=0)
        transceiver = client.addTransceiver("video", direction="recvonly")
        transceiver.setCodecPreferences([codec for codec in RTCRtpSender.getCapabilities("video").codecs if codec.mimeType == "video/H264"])
        incoming = asyncio.Queue()
        arrived = asyncio.Event()
        opened = asyncio.Event()
        metas, pongs, received_rtp, decoded, safari_decoded = [], [], [], [], []
        safari_incoming = asyncio.Queue()
        original_packet = transceiver.receiver._handle_rtp_packet

        async def received_packet(packet, arrival_time_ms):
            if packet.marker:
                received_rtp.append(packet.timestamp)
            return await original_packet(packet, arrival_time_ms)
        transceiver.receiver._handle_rtp_packet = received_packet

        @client.on("track")
        def track_ready(track):
            incoming.put_nowait(track)

        @safari.on("track")
        def safari_track_ready(track):
            safari_incoming.put_nowait(track)

        @channel.on("open")
        def channel_open():
            opened.set()
            channel.send(json.dumps({"type": "ping", "v": 1, "id": 1, "t0_ns": time.monotonic_ns()}))

        @channel.on("message")
        def channel_message(data):
            message = json.loads(data)
            (metas if message["type"] == "frame" else pongs).append(message)
            if len(metas) >= 12:
                arrived.set()

        async def request_json():
            return {"sdp": client.localDescription.sdp, "type": "offer", "codec": "h264", "video_metadata": "r1-video-meta-v1"}

        async def safari_json():
            return {"sdp": safari.localDescription.sdp, "type": "offer", "codec": "h264"}

        consumer = None
        safari_consumer = None
        try:
            await client.setLocalDescription(await client.createOffer())
            response = await publisher._offer(SimpleNamespace(json=request_json, headers={}))
            self.assertEqual(response.status, 200)
            answer = json.loads(response.text)
            await client.setRemoteDescription(RTCSessionDescription(**answer))
            await asyncio.wait_for(opened.wait(), 10)
            track = await asyncio.wait_for(incoming.get(), 5)
            safari.addTransceiver("video", direction="recvonly")
            await safari.setLocalDescription(await safari.createOffer())
            safari_response = await publisher._offer(SimpleNamespace(json=safari_json, headers={}))
            self.assertEqual(safari_response.status, 200)
            await safari.setRemoteDescription(RTCSessionDescription(**json.loads(safari_response.text)))
            safari_track = await asyncio.wait_for(safari_incoming.get(), 5)
            server_senders = [pc.getSenders()[0] for pc in publisher._pcs]
            self.assertEqual(sorted(sender.track._buffered for sender in server_senders), [False, True])
            self.assertEqual(sum("_next_encoded_frame" in sender.__dict__ for sender in server_senders), 1)

            async def consume():
                while True:
                    frame = await track.recv()
                    decoded.append((frame.pts, int(round(float(frame.to_ndarray(format="bgr24").mean())))))
            consumer = asyncio.create_task(consume())

            async def consume_safari():
                while True:
                    frame = await safari_track.recv()
                    safari_decoded.append(frame.to_ndarray(format="bgr24").shape)
            safari_consumer = asyncio.create_task(consume_safari())
            source_times = {}
            for seq in list(range(1, 9)) + [8, 8] + list(range(9, 15)):
                source_times.setdefault(seq, time.monotonic_ns())
                metadata = {"source_sequence": seq, "source_monotonic_ns": source_times[seq],
                            "clock_id": server.clock_id(), "source_epoch": "test-epoch",
                            "timestamp_kind": "pc2_rtp_decoded_receive"}
                publisher._bgr_track.push_frame((np.full((32, 64, 3), 40 + seq * 10, np.uint8), metadata), asyncio.get_running_loop())
                await asyncio.sleep(0.08)
            await asyncio.wait_for(arrived.wait(), 5)
            await asyncio.sleep(0.2)
            self.assertGreaterEqual(len(decoded), 10)
            self.assertGreaterEqual(len(safari_decoded), 10)
            self.assertEqual(set(safari_decoded), {(32, 64, 3)})
            self.assertTrue(pongs)
            self.assertEqual(pongs[0]["id"], 1)
            self.assertLessEqual(pongs[0]["receive_monotonic_ns"], pongs[0]["send_monotonic_ns"])
            meta_by_rtp = {meta["rtp_timestamp"]: meta for meta in metas}
            matched = 0
            for pts, pixel in decoded:
                rtp_timestamp = (received_rtp[0] + pts) & 0xFFFFFFFF
                meta = meta_by_rtp.get(rtp_timestamp)
                if meta is None:
                    continue
                self.assertIn(rtp_timestamp, received_rtp)
                seq = meta["source_sequence"]
                self.assertLessEqual(abs(pixel - (40 + seq * 10)), 4)
                self.assertEqual(meta["source_monotonic_ns"], source_times[seq])
                self.assertEqual(meta["source_epoch"], "test-epoch")
                matched += 1
            self.assertGreaterEqual(matched, 10)
            repeated = [meta for meta in metas if meta["source_sequence"] == 8]
            self.assertEqual(len(repeated), 3)
            self.assertEqual(len({meta["rtp_timestamp"] for meta in repeated}), 3)
            self.assertEqual(len({meta["source_monotonic_ns"] for meta in repeated}), 1)
            print(json.dumps({"actual_aiortc": "1.15.0", "rtp_markers": len(received_rtp), "metadata_frames": len(metas),
                              "decoded_frames": len(decoded), "exact_rtp_and_pixel_source_matches": matched,
                              "safari_simultaneous_decoded_frames": len(safari_decoded),
                              "repeated_source_frames_preserve_time": len(repeated), "pong_received": bool(pongs)}))
        finally:
            if consumer:
                consumer.cancel()
                await asyncio.gather(consumer, return_exceptions=True)
            if safari_consumer:
                safari_consumer.cancel()
                await asyncio.gather(safari_consumer, return_exceptions=True)
            await client.close()
            await safari.close()
            for pc in list(publisher._pcs):
                await publisher._cleanup_pc(pc)
            publisher._bgr_track.stop()


if __name__ == "__main__":
    unittest.main(verbosity=2)
