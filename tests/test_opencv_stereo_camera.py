import importlib.util
import logging
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


IMAGE_SERVER_PATH = Path(__file__).resolve().parents[1] / "src" / "teleimager" / "image_server.py"


class FakeBuffer:
    def __init__(self):
        self.value = None

    def write(self, value):
        self.value = value

    def read(self):
        return self.value


class FakeCapture:
    def __init__(self, name, frame, events):
        self.name = name
        self.frame = frame
        self.events = events
        self.properties = []
        self.released = False

    def set(self, prop, value):
        self.properties.append((prop, value))
        return True

    def isOpened(self):
        return True

    def grab(self):
        self.events.append((self.name, "grab"))
        return True

    def retrieve(self):
        self.events.append((self.name, "retrieve"))
        return True, self.frame.copy()

    def release(self):
        self.released = True


def load_image_server():
    cv2_module = types.ModuleType("cv2")
    cv2_module.CAP_V4L2 = 1
    cv2_module.CAP_PROP_FOURCC = 2
    cv2_module.CAP_PROP_FRAME_HEIGHT = 3
    cv2_module.CAP_PROP_FRAME_WIDTH = 4
    cv2_module.CAP_PROP_FPS = 5
    cv2_module.CAP_PROP_BUFFERSIZE = 6
    cv2_module.VideoWriter_fourcc = lambda *_args: 10
    cv2_module.VideoCapture = lambda *_args: None
    cv2_module.imencode = lambda _ext, _frame: (True, np.frombuffer(b"jpeg", dtype=np.uint8))

    h264_module = types.ModuleType("aiortc.codecs.h264")
    h264_module.MIN_BITRATE = 100_000
    h264_module.DEFAULT_BITRATE = 1_000_000
    h264_module.MAX_BITRATE = 2_000_000
    h264_module.H264Encoder = type("H264Encoder", (), {})
    vpx_module = types.ModuleType("aiortc.codecs.vpx")
    vpx_module.MIN_BITRATE = 100_000
    vpx_module.DEFAULT_BITRATE = 1_000_000
    vpx_module.MAX_BITRATE = 2_000_000
    codecs_module = types.ModuleType("aiortc.codecs")
    codecs_module.h264 = h264_module
    codecs_module.vpx = vpx_module

    aiortc_module = types.ModuleType("aiortc")
    aiortc_module.MediaStreamTrack = type("MediaStreamTrack", (), {})
    aiortc_module.RTCPeerConnection = type("RTCPeerConnection", (), {})
    aiortc_module.RTCSessionDescription = type("RTCSessionDescription", (), {})
    aiortc_module.RTCRtpSender = type("RTCRtpSender", (), {})
    media_module = types.ModuleType("aiortc.contrib.media")
    media_module.MediaRelay = type("MediaRelay", (), {})

    web = types.SimpleNamespace(
        Application=type("Application", (), {}),
        AppRunner=type("AppRunner", (), {}),
        Request=type("Request", (), {}),
        Response=type("Response", (), {}),
        TCPSite=type("TCPSite", (), {}),
    )
    aiohttp_module = types.ModuleType("aiohttp")
    aiohttp_module.web = web

    video_frame = type("VideoFrame", (), {})
    av_module = types.ModuleType("av")
    av_module.VideoFrame = video_frame
    av_module.video = types.SimpleNamespace(
        frame=types.SimpleNamespace(PictureType=types.SimpleNamespace(I=1, NONE=0))
    )

    image_client_module = types.ModuleType("teleimager.image_client")
    image_client_module.TripleRingBuffer = FakeBuffer
    image_client_module.ZMQ_PublisherManager = type(
        "ZMQ_PublisherManager", (), {"get_instance": classmethod(lambda cls: cls())}
    )
    image_client_module.ZMQ_Responser = type(
        "ZMQ_Responser", (), {"__init__": lambda self, config: None, "stop": lambda self: None}
    )
    teleimager_module = types.ModuleType("teleimager")
    teleimager_module.__path__ = [str(IMAGE_SERVER_PATH.parent)]

    fake_modules = {
        "logging_mp": logging,
        "cv2": cv2_module,
        "yaml": types.ModuleType("yaml"),
        "aiohttp": aiohttp_module,
        "aiortc": aiortc_module,
        "aiortc.codecs": codecs_module,
        "aiortc.codecs.h264": h264_module,
        "aiortc.codecs.vpx": vpx_module,
        "aiortc.contrib": types.ModuleType("aiortc.contrib"),
        "aiortc.contrib.media": media_module,
        "aiortc.rtcrtpsender": aiortc_module,
        "av": av_module,
        "teleimager": teleimager_module,
        "teleimager.image_client": image_client_module,
    }
    module_patch = mock.patch.dict(sys.modules, fake_modules)
    module_patch.start()
    spec = importlib.util.spec_from_file_location("teleimager.image_server_under_test", IMAGE_SERVER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, module_patch


class OpenCVStereoCameraTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.image_server, cls.module_patch = load_image_server()

    @classmethod
    def tearDownClass(cls):
        sys.modules.pop("teleimager.image_server_under_test", None)
        cls.module_patch.stop()

    def make_captures(self):
        events = []
        captures = {
            "/dev/video2": FakeCapture("left", np.full((480, 640, 3), 10, dtype=np.uint8), events),
            "/dev/video4": FakeCapture("right", np.full((480, 640, 3), 20, dtype=np.uint8), events),
        }
        return captures, events

    def test_grabs_both_before_retrieve_and_outputs_left_then_right(self):
        captures, events = self.make_captures()
        with mock.patch.object(self.image_server.cv2, "VideoCapture", side_effect=lambda path, _backend: captures[path]):
            camera = self.image_server.OpenCVStereoCamera(
                "head_camera", "/dev/video2", "/dev/video4", [480, 1280], 30,
                enable_zmq=True, enable_webrtc=True,
            )
            camera._update_frame()

        self.assertEqual(
            events,
            [("left", "grab"), ("right", "grab"), ("left", "retrieve"), ("right", "retrieve")],
        )
        frame = camera.get_bgr_frame()
        self.assertEqual(frame.shape, (480, 1280, 3))
        np.testing.assert_array_equal(frame[:, :640], captures["/dev/video2"].frame)
        np.testing.assert_array_equal(frame[:, 640:], captures["/dev/video4"].frame)
        self.assertEqual(camera.get_jpeg_bytes(), b"jpeg")

        camera.release()
        self.assertTrue(captures["/dev/video2"].released)
        self.assertTrue(captures["/dev/video4"].released)

    def test_resolves_both_physical_paths(self):
        captures, _events = self.make_captures()

        class FakeFinder:
            calls = []

            def __init__(self, *_args):
                pass

            def get_vpath_by_ppath(self, physical_path):
                self.calls.append(physical_path)
                return {"/sys/left": "/dev/video2", "/sys/right": "/dev/video4"}[physical_path]

        config = {
            "head_camera": {
                "enable_zmq": True,
                "zmq_port": 55555,
                "enable_webrtc": True,
                "webrtc_port": 60001,
                "type": "opencv_stereo",
                "image_shape": [480, 1280],
                "fps": 30,
                "left_physical_path": "/sys/left",
                "right_physical_path": "/sys/right",
            }
        }
        with mock.patch.object(self.image_server, "CameraFinder", FakeFinder), mock.patch.object(
            self.image_server.cv2, "VideoCapture", side_effect=lambda path, _backend: captures[path]
        ):
            server = self.image_server.ImageServer(config)

        camera = server._cameras["head_camera"]
        self.assertEqual(FakeFinder.calls, ["/sys/left", "/sys/right"])
        camera.release()


if __name__ == "__main__":
    unittest.main()
