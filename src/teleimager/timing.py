"""JPEG source timestamps and round-trip mapping between two monotonic clocks."""
from collections import deque
import json
from pathlib import Path
import struct

MAGIC = b"teleimager-time-v1\0"
CLOCK_MAX_AGE_NS = 10_000_000_000


def clock_id():
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def timestamp_jpeg(jpeg, metadata):
    # APP15 keeps each ZMQ message a valid JPEG, including for direct JPEG consumers.
    data = MAGIC + json.dumps(metadata, separators=(',', ':'), allow_nan=False).encode()
    return jpeg[:2] + b'\xff\xef' + struct.pack('>H', len(data) + 2) + data + jpeg[2:]


def jpeg_timestamp(jpeg):
    if jpeg[:4] != b'\xff\xd8\xff\xef' or jpeg[6:6 + len(MAGIC)] != MAGIC:
        return None
    if len(jpeg) < 6:
        return None
    length = struct.unpack('>H', jpeg[4:6])[0]
    try:
        data = json.loads(jpeg[6 + len(MAGIC):4 + length])
        if not isinstance(data, dict) or not isinstance(data.get('clock_id'), str):
            return None
        if any(type(data.get(key)) is not int or data[key] <= 0
               for key in ('source_monotonic_ns', 'source_sequence')):
            return None
        return data
    except (ValueError, UnicodeDecodeError):
        return None


class ClockMapping:
    def __init__(self):
        self.samples = deque(maxlen=16)

    def observe(self, sent_ns, received_ns, reply):
        if not isinstance(reply, dict) or not isinstance(reply.get('clock_id'), str):
            return False
        remote_receive, remote_send = reply.get('receive_ns'), reply.get('send_ns')
        if any(type(value) is not int for value in (remote_receive, remote_send)):
            return False
        delay = received_ns - sent_ns - (remote_send - remote_receive)
        if remote_send < remote_receive or received_ns < sent_ns or delay < 0:
            return False
        offset = ((sent_ns - remote_receive) + (received_ns - remote_send)) // 2
        self.samples.append({'clock_id': reply['clock_id'], 'measured_ns': received_ns,
                             'offset_ns': offset, 'network_uncertainty_ns': (delay + 1) // 2})
        return True

    def map(self, metadata, now_ns):
        if metadata is None:
            return {'clock_valid': False, 'timestamp_kind': 'host_receive_only'}
        result = dict(metadata)
        candidates = [sample for sample in self.samples
                      if sample['clock_id'] == metadata['clock_id']
                      and 0 <= now_ns - sample['measured_ns'] <= CLOCK_MAX_AGE_NS]
        if not candidates:
            result['clock_valid'] = False
            return result
        best = min(candidates, key=lambda sample: sample['network_uncertainty_ns'])
        age_ns = now_ns - best['measured_ns']
        mapped = metadata['source_monotonic_ns'] + best['offset_ns']
        # This is a drift allowance, not a measured hardware accuracy guarantee.
        uncertainty = best['network_uncertainty_ns'] + age_ns // 10_000  # 100 ppm
        result.update(mapped_monotonic_ns=mapped, clock_offset_ns=best['offset_ns'],
                      clock_uncertainty_ns=uncertainty, clock_age_ns=age_ns,
                      clock_measured_monotonic_ns=best['measured_ns'],
                      clock_network_uncertainty_ns=best['network_uncertainty_ns'],
                      clock_valid=mapped <= now_ns + uncertainty)
        return result
