import argparse
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import threading
import time


# Local receive-time approximation, not synchronized sensor exposure timestamps.
#
# Left and right are the two halves of one stereo sensor, so a matched pair has
# nominally identical capture times and any difference is transport jitter. A
# pair further apart than MAX_PAIR_SKEW belongs to two different sensor frames
# and is worth dropping. The window therefore has to sit *below* one frame
# period: at 15 fps (66.7 ms) 40 ms was safe, but the head stereo was raised to
# 30 fps (33.3 ms) on 2026-09-15, where 40 ms can no longer tell a matched pair
# from an adjacent one.
MAX_PAIR_SKEW = float(os.environ.get("R1_STEREO_MAX_PAIR_SKEW", "0.040"))
MAX_FRAME_AGE = float(os.environ.get("R1_STEREO_MAX_FRAME_AGE", "0.250"))
# How often take() reports what it has been doing, in seconds. 0 disables it.
STATS_INTERVAL = float(os.environ.get("R1_STEREO_STATS_INTERVAL", "5.0"))
HEADER = struct.Struct("!4sBIIIdI")


class StereoFrames:
    def __init__(self):
        self.condition = threading.Condition()
        self.frames = [None, None]
        self.last_received = [None, None]
        self.error = None
        self.stopped = False
        self.max_pair_skew = MAX_PAIR_SKEW
        self.pairs = 0
        self.rejections = 0
        self.timeouts = 0
        self.pair_skews = []
        self.rejected_skews = []
        self.stats_at = time.monotonic()

    @staticmethod
    def _ms(value):
        return "n/a" if value is None else f"{1000.0 * value:.1f}"

    @staticmethod
    def _quantile(values, fraction):
        if not values:
            return None
        ordered = sorted(values)
        return ordered[min(len(ordered) - 1, int(fraction * (len(ordered) - 1)))]

    def _report_stats(self, now, force=False):
        """Emit one line per interval so the pairing behaviour stays observable.

        Called with the condition held, so the counters it reads and resets
        cannot be touched concurrently.
        """
        if STATS_INTERVAL <= 0 or (not force and now - self.stats_at < STATS_INTERVAL):
            return
        elapsed = now - self.stats_at
        if self.pairs or self.rejections or self.timeouts:
            print(
                f"[GStreamer stereo] {elapsed:.1f}s pairs={self.pairs} "
                f"rejected_skew={self.rejections} timeouts={self.timeouts} "
                f"reject_rate={100.0 * self.rejections / max(1, self.pairs + self.rejections):.2f}% | "
                f"pair skew ms p50={self._ms(self._quantile(self.pair_skews, 0.50))} "
                f"p95={self._ms(self._quantile(self.pair_skews, 0.95))} "
                f"max={self._ms(max(self.pair_skews) if self.pair_skews else None)} | "
                f"rejected skew ms min={self._ms(min(self.rejected_skews) if self.rejected_skews else None)} "
                f"p50={self._ms(self._quantile(self.rejected_skews, 0.50))}",
                file=sys.stderr, flush=True,
            )
        self.pairs = 0
        self.rejections = 0
        self.timeouts = 0
        self.pair_skews = []
        self.rejected_skews = []
        self.stats_at = now

    def put(self, eye, timestamp, width, height, stride, data):
        with self.condition:
            self.frames[eye] = (timestamp, width, height, stride, data)
            self.last_received[eye] = timestamp
            self.condition.notify_all()

    def take(self, timeout):
        deadline = time.monotonic() + timeout
        with self.condition:
            while True:
                if self.stopped:
                    raise RuntimeError("Stereo capture stopped")
                if self.error:
                    raise RuntimeError(self.error)
                now = time.monotonic()
                self._report_stats(now)
                for eye, frame in enumerate(self.frames):
                    if frame is not None and now - frame[0] > MAX_FRAME_AGE:
                        self.frames[eye] = None
                left, right = self.frames
                if left is not None and right is not None:
                    skew = abs(left[0] - right[0])
                    if skew <= self.max_pair_skew:
                        self.frames = [None, None]
                        self.pairs += 1
                        self.pair_skews.append(skew)
                        return left, right
                    # Further apart than a matched pair can be: drop the older
                    # eye and resynchronise on the next one from that side.
                    self.rejections += 1
                    self.rejected_skews.append(skew)
                    self.frames[0 if left[0] < right[0] else 1] = None
                remaining = deadline - now
                if remaining <= 0:
                    self.timeouts += 1
                    self._report_stats(now, force=True)
                    ages = ["never" if stamp is None else f"{now - stamp:.3f}s"
                            for stamp in self.last_received]
                    raise TimeoutError(
                        f"No fresh stereo pair: left age={ages[0]}, right age={ages[1]}; "
                        f"receive-time skew limit={self.max_pair_skew:.3f}s, "
                        f"age limit={MAX_FRAME_AGE:.3f}s"
                    )
                self.condition.wait(remaining)


class StereoCapture:
    def __init__(self, left_port, right_port):
        self.frames = StereoFrames()
        self.process = subprocess.Popen(
            ["/usr/bin/python3", "-u", str(Path(__file__).resolve()),
             "--left-port", str(int(left_port)), "--right-port", str(int(right_port))],
            stdout=subprocess.PIPE, bufsize=0,
            # Inherit stderr so GStreamer errors reach the service journal.
        )
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read_exact(self, size):
        data = bytearray()
        while len(data) < size:
            chunk = self.process.stdout.read(size - len(data))
            if not chunk:
                raise EOFError("GStreamer stereo helper exited; see preceding decoder log")
            data.extend(chunk)
        return bytes(data)

    def _read(self):
        try:
            while True:
                magic, eye, width, height, stride, timestamp, size = HEADER.unpack(self._read_exact(HEADER.size))
                if (magic != b"R1ST" or eye > 1 or not 0 < width <= 8192
                        or not 0 < height <= 8192 or stride < width * 3
                        or size != height * stride or size > 128 * 1024 * 1024):
                    raise ValueError("Invalid raw frame header from GStreamer stereo helper")
                self.frames.put(eye, timestamp, width, height, stride, self._read_exact(size))
        except Exception as error:
            with self.frames.condition:
                self.frames.error = str(error)
                self.frames.condition.notify_all()

    def close(self):
        with self.frames.condition:
            self.frames.stopped = True
            self.frames.condition.notify_all()
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=1.0)
        self.reader.join(timeout=1.0)
        self.process.stdout.close()


def run_capture(left_port, right_port, test_source=False):
    import gi
    gi.require_version("Gst", "1.0")
    gi.require_version("GstVideo", "1.0")
    from gi.repository import Gst, GstVideo

    Gst.init(None)
    branches = []
    for eye, port in enumerate((left_port, right_port)):
        eye_name = ("left", "right")[eye]
        source = (f"videotestsrc is-live=true pattern={eye} ! video/x-raw,width=544,height=448,framerate=15/1"
                  if test_source else
                  f"udpsrc name={eye_name}_source port={port} do-timestamp=true "
                  'caps="application/x-rtp,media=video,clock-rate=90000,encoding-name=H264,payload=96" '
                  f"! rtph264depay name={eye_name}_depay ! h264parse ! avdec_h264 name={eye_name}_decoder")
        branches.append(
            source + " ! queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 leaky=downstream "
            f"! videoconvert ! video/x-raw,format=BGR ! appsink name=eye{eye} "
            "emit-signals=true max-buffers=1 drop=true sync=false async=false"
        )
    pipeline = Gst.parse_launch(" ".join(branches))
    condition = threading.Condition()
    latest = [None, None]
    errors = []
    stopped = threading.Event()

    def stop(signum, frame):
        stopped.set()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def sample_ready(sink, eye):
        try:
            sample = sink.emit("pull-sample")
            buffer = sample.get_buffer()
            info = GstVideo.VideoInfo()
            if not info.from_caps(sample.get_caps()):
                raise RuntimeError(f"eye {eye}: invalid BGR caps")
            running_pts = sample.get_segment().to_running_time(Gst.Format.TIME, buffer.pts)
            if running_pts == Gst.CLOCK_TIME_NONE:
                raise RuntimeError(f"eye {eye}: missing local receive timestamp")
            running_now = pipeline.get_clock().get_time() - pipeline.get_base_time()
            timestamp = time.monotonic() - max(0, running_now - running_pts) / Gst.SECOND
            data = buffer.extract_dup(0, info.stride[0] * info.height)
            packet = HEADER.pack(b"R1ST", eye, info.width, info.height, info.stride[0], timestamp, len(data)) + data
            with condition:
                latest[eye] = packet
                condition.notify()
            return Gst.FlowReturn.OK
        except Exception as error:
            with condition:
                errors.append(str(error))
                condition.notify()
            return Gst.FlowReturn.ERROR

    for eye in range(2):
        pipeline.get_by_name(f"eye{eye}").connect("new-sample", sample_ready, eye)
    try:
        if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("GStreamer stereo pipeline failed to start")
        bus = pipeline.get_bus()
        while not stopped.is_set():
            message = bus.pop_filtered(Gst.MessageType.ERROR | Gst.MessageType.WARNING | Gst.MessageType.EOS)
            if message is not None:
                if message.type == Gst.MessageType.ERROR:
                    error, debug = message.parse_error()
                    raise RuntimeError(f"{message.src.get_name()}: {error}; {debug}")
                if message.type == Gst.MessageType.EOS:
                    raise RuntimeError("GStreamer stereo pipeline ended")
                warning, debug = message.parse_warning()
                print(f"[GStreamer stereo] {message.src.get_name()}: {warning}; {debug}", file=sys.stderr)
            with condition:
                if errors:
                    raise RuntimeError(errors[0])
                packets = latest[:]
                latest[:] = [None, None]
                if not any(packets):
                    condition.wait(0.050)
            for packet in packets:
                if packet is not None:
                    # The callbacks keep replacing one slot per eye if this pipe is slow.
                    remaining = memoryview(packet)
                    while remaining:
                        remaining = remaining[os.write(sys.stdout.fileno(), remaining):]
    finally:
        pipeline.set_state(Gst.State.NULL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-port", type=int, default=5002)
    parser.add_argument("--right-port", type=int, default=5003)
    parser.add_argument("--test-source", action="store_true")
    parser.add_argument("--max-pair-skew", type=float, default=None,
                        help="Override MAX_PAIR_SKEW (seconds) in this process only")
    args = parser.parse_args()
    if args.max_pair_skew is not None:
        MAX_PAIR_SKEW = args.max_pair_skew
    try:
        run_capture(args.left_port, args.right_port, args.test_source)
    except Exception as error:
        print(f"[GStreamer stereo] {error}", file=sys.stderr)
        sys.exit(1)
