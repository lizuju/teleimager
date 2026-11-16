import asyncio
import importlib.util
import json
import logging
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import av
import numpy as np
import teleimager
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration
from aiortc.contrib.media import MediaRelay

ROOT = Path(__file__).resolve().parents[1]
teleimager.__path__.insert(0, str(ROOT / "src/teleimager"))
spec = importlib.util.spec_from_file_location("teleimager.server_under_test", ROOT / "src/teleimager/server.py")
server = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {"logging_mp": logging}):
    spec.loader.exec_module(server)


class Request:
    def __init__(self, client="headset-a", native=True, remote="gateway", sdp="offer"):
        self.headers = {}
        self.remote = remote
        self.params = {"sdp": sdp, "type": "offer"}
        if native:
            self.params["video_metadata"] = server.CHANNEL_LABEL
        if client is not None:
            self.params["native_client_id"] = client

    async def json(self):
        return self.params


class FakePeer:
    def __init__(self, gate=None, fail=False):
        self.gate, self.fail = gate, fail
        self.started = asyncio.Event()
        self.connectionState = "new"
        self.localDescription = SimpleNamespace(sdp="answer", type="answer")
        self.handlers = {}
        self.track = None

    def on(self, event, callback=None):
        def register(fn):
            self.handlers[event] = fn
            return fn
        return register(callback) if callback else register

    def addTransceiver(self, track, direction):
        self.track = track
        return SimpleNamespace(sender=object(), setCodecPreferences=lambda codecs: None)

    async def setRemoteDescription(self, offer):
        self.started.set()
        if self.gate is not None:
            await self.gate.wait()
        if self.fail:
            raise ValueError("bad SDP")

    async def createAnswer(self):
        return self.localDescription

    async def setLocalDescription(self, answer):
        pass

    async def close(self):
        self.connectionState = "closed"


class ReplacementTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.publisher = server.WebRTC_PublisherThread(0)
        self.publisher._bgr_track = server.BGRArrayVideoStreamTrack()
        self.publisher._relay = MediaRelay()
        self.metadata = patch.object(server, "VideoMetadataSender", return_value=SimpleNamespace(attach=lambda _: None))
        self.metadata.start()

    async def asyncTearDown(self):
        await self.publisher._shutdown()
        self.publisher._bgr_track.stop()
        self.metadata.stop()

    async def offer(self, request=None, peer=None):
        peer = peer or FakePeer()
        with patch.object(server, "RTCPeerConnection", return_value=peer):
            response = await self.publisher._offer(request or Request())
        return peer, response

    async def test_replaces_only_matching_native_client(self):
        old, _ = await self.offer()
        other, _ = await self.offer(Request("headset-b"))
        safari, _ = await self.offer(Request("headset-a", native=False))
        direct, _ = await self.offer(Request("headset-a", remote="other-gateway"))
        new, response = await self.offer()
        self.assertEqual(response.status, 200)
        self.assertEqual(old.connectionState, "closed")
        self.assertEqual(old.track.readyState, "ended")
        self.assertEqual(self.publisher._pcs, {new, other, safari, direct})
        self.assertEqual(self.publisher._bgr_track.readyState, "live")
        self.assertEqual(other.track.readyState, "live")
        await old.handlers["connectionstatechange"]()
        self.assertIs(self.publisher._native_peers[("gateway", "headset-a")], new)

    async def test_failed_negotiation_preserves_existing_peer(self):
        old, _ = await self.offer()
        failed, response = await self.offer(peer=FakePeer(fail=True))
        self.assertEqual(response.status, 400)
        self.assertEqual(self.publisher._pcs, {old})
        self.assertEqual(failed.track.readyState, "ended")
        self.assertIs(self.publisher._native_peers[("gateway", "headset-a")], old)
        self.assertFalse(self.publisher._native_pending)

    async def test_late_old_offer_cannot_replace_newer_offer(self):
        original, _ = await self.offer()
        gate = asyncio.Event()
        slow, fast = FakePeer(gate=gate), FakePeer()
        with patch.object(server, "RTCPeerConnection", side_effect=[slow, fast]):
            pending = asyncio.create_task(self.publisher._offer(Request()))
            await slow.started.wait()
            response = await self.publisher._offer(Request())
            gate.set()
            superseded = await pending
        self.assertEqual(response.status, 200)
        self.assertEqual(superseded.status, 409)
        self.assertEqual(self.publisher._pcs, {fast})
        self.assertEqual(slow.track.readyState, "ended")
        self.assertEqual(original.connectionState, "closed")

    async def test_failed_latest_offer_does_not_revive_pending_old_offer(self):
        original, _ = await self.offer()
        gate = asyncio.Event()
        slow, failed = FakePeer(gate=gate), FakePeer(fail=True)
        with patch.object(server, "RTCPeerConnection", side_effect=[slow, failed]):
            pending = asyncio.create_task(self.publisher._offer(Request()))
            await slow.started.wait()
            self.assertEqual((await self.publisher._offer(Request())).status, 400)
            gate.set()
            self.assertEqual((await pending).status, 409)
        self.assertEqual(self.publisher._pcs, {original})

    async def test_cancelled_offer_releases_track_and_preserves_old_peer(self):
        old, _ = await self.offer()
        candidate = FakePeer(gate=asyncio.Event())
        with patch.object(server, "RTCPeerConnection", return_value=candidate):
            pending = asyncio.create_task(self.publisher._offer(Request()))
            await candidate.started.wait()
            pending.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await pending
        self.assertEqual(self.publisher._pcs, {old})
        self.assertEqual(candidate.track.readyState, "ended")
        self.assertFalse(self.publisher._native_pending)

    async def test_replacement_during_old_close_does_not_return_closed_answer(self):
        original, _ = await self.offer()
        closing, release = asyncio.Event(), asyncio.Event()

        async def slow_close():
            closing.set()
            await release.wait()
            original.connectionState = "closed"

        original.close = slow_close
        middle, latest = FakePeer(), FakePeer()
        with patch.object(server, "RTCPeerConnection", side_effect=[middle, latest]):
            pending = asyncio.create_task(self.publisher._offer(Request()))
            await closing.wait()
            response = await self.publisher._offer(Request())
            release.set()
            superseded = await pending
        self.assertEqual(response.status, 200)
        self.assertEqual(superseded.status, 409)
        self.assertEqual(self.publisher._pcs, {latest})
        self.assertEqual(middle.track.readyState, "ended")

    async def test_cancel_during_replacement_finishes_closing_both_peers(self):
        original, _ = await self.offer()
        closing, release, closed = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def slow_close():
            closing.set()
            await release.wait()
            original.connectionState = "closed"
            closed.set()

        original.close = slow_close
        new = FakePeer()
        with patch.object(server, "RTCPeerConnection", return_value=new):
            pending = asyncio.create_task(self.publisher._offer(Request()))
            await closing.wait()
            pending.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await pending
            release.set()
            await asyncio.wait_for(closed.wait(), 1)
        self.assertEqual(new.connectionState, "closed")
        self.assertFalse(self.publisher._pcs)
        self.assertFalse(self.publisher._native_peers)
        self.assertFalse(self.publisher._peer_tracks)

    async def test_legacy_native_and_safari_remain_independent(self):
        a, _ = await self.offer(Request(None))
        b, _ = await self.offer(Request(None))
        safari, _ = await self.offer(Request(None, native=False))
        self.assertEqual(self.publisher._pcs, {a, b, safari})

    async def test_invalid_identity_rejected_before_allocating_peer(self):
        old, _ = await self.offer()
        for value in ([], 12, "", "x" * 129):
            _, response = await self.offer(Request(value))
            self.assertEqual(response.status, 400)
        self.assertEqual(self.publisher._pcs, {old})

    async def test_shutdown_releases_pending_peers_and_tracks(self):
        peer, _ = await self.offer()
        await self.publisher._shutdown()
        self.assertFalse(self.publisher._pcs)
        self.assertFalse(self.publisher._native_peers)
        self.assertFalse(self.publisher._native_pending)
        self.assertFalse(self.publisher._peer_tracks)
        self.assertEqual(peer.track.readyState, "ended")


class RealPeerTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_video_survives_replacement_with_safari_connected(self):
        publisher = server.WebRTC_PublisherThread(0)
        publisher._bgr_track = server.BGRArrayVideoStreamTrack()
        publisher._relay = MediaRelay()
        clients = []
        tracks = []
        stopped = asyncio.Event()
        real_factory = RTCPeerConnection

        async def frames():
            while not stopped.is_set():
                publisher._bgr_track.push_frame(np.zeros((16, 32, 3), dtype=np.uint8), asyncio.get_running_loop())
                await asyncio.sleep(1 / 30)

        async def connect(name, native=True):
            client = real_factory(RTCConfiguration(iceServers=[]))
            clients.append(client)
            incoming = asyncio.Future()
            client.on("track", lambda track: incoming.set_result(track))
            client.addTransceiver("video", direction="recvonly")
            if native:
                client.createDataChannel(server.CHANNEL_LABEL, ordered=False, maxRetransmits=0)
            await client.setLocalDescription(await client.createOffer())
            request = Request(name, native=native, sdp=client.localDescription.sdp)
            with patch.object(server, "RTCPeerConnection", side_effect=lambda: real_factory(RTCConfiguration(iceServers=[]))):
                response = await publisher._offer(request)
            self.assertEqual(response.status, 200)
            answer = json.loads(response.text)
            await client.setRemoteDescription(RTCSessionDescription(**answer))
            track = await asyncio.wait_for(incoming, 5)
            tracks.append(track)
            await asyncio.wait_for(track.recv(), 5)
            return client, track

        producer = asyncio.create_task(frames())
        try:
            old_client, _ = await connect("headset-a")
            old = publisher._native_peers[("gateway", "headset-a")]
            old_track = publisher._peer_tracks[old]
            _, safari = await connect(None, native=False)
            for _ in range(3):
                _, newest = await connect("headset-a")
                self.assertEqual(len(publisher._pcs), 2)
                self.assertEqual(len(publisher._peer_tracks), 2)
                await asyncio.wait_for(newest.recv(), 5)
                await asyncio.wait_for(safari.recv(), 5)
            self.assertEqual(old.connectionState, "closed")
            self.assertEqual(old_track.readyState, "ended")
            self.assertEqual(publisher._bgr_track.readyState, "live")
        finally:
            stopped.set()
            await producer
            await asyncio.gather(*(client.close() for client in clients))
            await publisher._shutdown()
            publisher._bgr_track.stop()


if __name__ == "__main__":
    unittest.main(verbosity=2)
