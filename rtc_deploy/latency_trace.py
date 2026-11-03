"""Nonblocking application trace hooks; ROS 2 runs only in the collector.

No robot messages are published here. UDP telemetry may be lost; gaps stay gaps.
Enable: LATENCY_TRACE_UDP=127.0.0.1:19555 LATENCY_TRACE_HOST=229
"""
import argparse
import collections
import json
import math
import os
from pathlib import Path
import socket
import sys
import time
import uuid

HOST = os.environ.get('LATENCY_TRACE_HOST', socket.gethostname())
try:
    BOOT = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
except OSError:
    BOOT = str(uuid.uuid4())
_sock = None
_addr = None
_enabled = bool(os.environ.get('LATENCY_TRACE_UDP'))
DROPPED = 0


def stamp():
    # Epoch clock is used only with explicit external synchronization evidence.
    return dict(host=HOST, boot=BOOT, mono_ns=time.monotonic_ns(), wall_ns=time.time_ns())


def mark(trace_id, stage, at=None, **meta):
    """Best-effort local UDP; never throw into control/inference code."""
    global _sock, _addr, DROPPED
    if not _enabled:
        return
    try:
        event = dict(v=1, event_id=uuid.uuid4().hex, trace_id=str(trace_id),
                     stage=stage, clock=at or stamp(), meta=meta, emitter_drops=DROPPED)
        if _sock is None:
            host, port = os.environ['LATENCY_TRACE_UDP'].rsplit(':', 1)
            # Keep the producer local; forwarding belongs in a separate process.
            if host != '127.0.0.1':
                raise ValueError('Trace producer must target 127.0.0.1')
            _addr = (host, int(port))
            _sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            _sock.setblocking(False)
        wire = json.dumps(event, separators=(',', ':'), allow_nan=False).encode()
        if len(wire) > 8192:
            raise ValueError('Trace event too large')
        _sock.sendto(wire, _addr)
    except Exception:
        DROPPED += 1


# Each pair is a measurement boundary, not necessarily pure wire latency.
EDGES = [
    ('camera_request_roundtrip_229', 'camera_request_229', 'camera_payload_229'),
    ('camera_jpeg_decode_229', 'camera_payload_229', 'camera_jpeg_decoded_229'),
    ('g1_camera_wait', 'g1_camera_requested', 'g1_camera_selected'),
    ('g1_camera_frame_age', 'g1_mjpeg_ready', 'g1_camera_selected'),
    ('g1_camera_send_api', 'g1_camera_send', 'g1_camera_send_done'),
    ('camera_g1_to_229_response', 'g1_camera_send', 'camera_payload_229'),
    ('rtc_to_sonic_token_decode_229', 'action_send', 'sonic_token_decoded'),
    ('camera_g1_to_229', 'g1_camera_send', 'camera_229_receive'),
    ('camera_decode_229', 'camera_229_receive', 'camera_229_decoded'),
    ('observation_build_229', 'obs_build_start', 'obs_ready'),
    ('observation_wait_229', 'obs_ready', 'request_queued'),
    ('request_queue_229', 'request_queued', 'request_encode_start'),
    ('request_json_229', 'request_encode_start', 'request_send_start'),
    ('request_send_api_229', 'request_send_start', 'request_send_done'),
    ('uplink_229_to_32', 'request_send_start', 'server_received'),
    ('request_json_32', 'server_received', 'server_decoded'),
    ('server_wait_32', 'server_decoded', 'predict_start'),
    ('sonic_decoder_229', 'sonic_decode_start', 'sonic_decode_done'),
    ('sonic_command_prepare_229', 'sonic_decode_done', 'sonic_command_ready'),
    ('sonic_command_queue_229', 'dds_command_ready', 'dds_write_start'),
    ('dds_write_api_229', 'dds_write_start', 'dds_write_done'),
    ('preprocess_32', 'predict_start', 'preprocess_done'),
    ('model_and_postprocess_32', 'preprocess_done', 'predict_done'),
    ('predict_total_32', 'predict_start', 'predict_done'),
    ('server_reply_prepare_32', 'predict_done', 'reply_send_start'),
    ('downlink_32_to_229', 'reply_send_start', 'reply_received'),
    ('reply_json_229', 'reply_received', 'reply_decoded'),
    ('reply_accept_229', 'reply_decoded', 'reply_accepted'),
    ('request_roundtrip_229', 'request_send_start', 'reply_received'),
    ('action_queue_229', 'action_ready', 'action_send'),
    ('rtc_to_sonic_229', 'action_send', 'sonic_received'),
    ('sonic_to_dds_write_229', 'sonic_received', 'dds_write'),
    ('dds_229_to_g1', 'dds_write', 'g1_action_received'),
    ('g1_receiver_to_use', 'g1_action_received', 'g1_action_used'),
]


def clock_intervals(events):
    samples = collections.defaultdict(list)
    for e in events:
        if e.get('stage') == 'clock_sync':
            c = e['clock']
            samples[(c['host'], c['boot'])].append(e)
    result = collections.defaultdict(list)
    for (host, boot), records in samples.items():
        records.sort(key=lambda e: e['clock']['mono_ns'])
        for left, right in zip(records, records[1:]):
            lm, rm = left['meta'], right['meta']
            if lm.get('status') != 'ok' or rm.get('status') != 'ok':
                continue
            le, re = lm['estimate'], rm['estimate']
            if le.get('reference') != re.get('reference'):
                continue
            lc, rc = left['clock'], right['clock']
            seconds = (rc['mono_ns'] - lc['mono_ns']) / 1e9
            if not 0 < seconds <= 10:
                continue
            discontinuity_ms = abs((rc['wall_ns']-lc['wall_ns']) - (rc['mono_ns']-lc['mono_ns']))/1e6
            if discontinuity_ms > 1:
                continue
            uncertainty = max(le['uncertainty_ms'], re['uncertainty_ms'])
            query_ms = max((e['clock']['mono_ns']-e['meta']['query_start']['mono_ns'])/1e6
                           for e in (left, right))
            uncertainty += query_ms + discontinuity_ms + seconds*max(le['skew_ppm'],re['skew_ppm'])/1000
            if not math.isfinite(uncertainty) or uncertainty < 0:
                continue
            result[host].append(dict(boot=boot, offset_ms=0, uncertainty_ms=uncertainty,
                                     valid_from_ns=lc['wall_ns'],valid_until_ns=rc['wall_ns'],
                                     source='chrony_bracketed', reference_scale='UTC'))
    return result


def select_calibration(clock, sync):
    entries = sync[clock['host']]
    if isinstance(entries, dict):
        entries = [entries]  # Legacy explicit calibration file.
    for entry in entries:
        if entry.get('boot', clock['boot']) != clock['boot']:
            continue
        if entry['valid_from_ns'] <= clock['wall_ns'] <= entry['valid_until_ns']:
            return entry
    if isinstance(sync[clock['host']], dict):
        return sync[clock['host']]
    raise KeyError('No clock evidence covering event')


def duration(a, b, sync):
    ca, cb = a['clock'], b['clock']
    if ca['host'] == cb['host'] and ca['boot'] == cb['boot']:
        ms = (cb['mono_ns'] - ca['mono_ns']) / 1e6
        return (ms, 0.0, 'monotonic') if ms >= 0 else (None, None, 'invalid_order')
    # The user supplies measured bounds, NOT a switch asserting clocks are equal.
    try:
        sa, sb = select_calibration(ca, sync), select_calibration(cb, sync)
        for c, s in ((ca, sa), (cb, sb)):
            if not s['valid_from_ns'] <= c['wall_ns'] <= s['valid_until_ns']:
                return None, None, 'sync_evidence_expired'
            if s['uncertainty_ms'] < 0 or not math.isfinite(s['uncertainty_ms']):
                return None, None, 'invalid_sync_evidence'
        # offset_ms = host realtime minus common reference realtime.
        ms = (cb['wall_ns']-ca['wall_ns'])/1e6 - sb['offset_ms'] + sa['offset_ms']
        error = sa['uncertainty_ms'] + sb['uncertainty_ms']
        if not math.isfinite(ms):
            return None, None, 'invalid_sync_evidence'
        return (ms, error, 'synced_epoch') if ms >= 0 else (None, error, 'negative_clock_delta')
    except (KeyError, TypeError):
        return None, None, 'unsynchronized'


def analyze(events, sync=None):
    events = list(events)
    if sync is None:
        sync = clock_intervals(events)
    groups = collections.defaultdict(dict)
    duplicate_keys = set()
    seen = set()
    for e in events:
        if e['event_id'] in seen:
            continue
        seen.add(e['event_id'])
        key = (e['trace_id'], e['stage'])
        if e['stage'] in groups[e['trace_id']]:
            duplicate_keys.add(key)
        groups[e['trace_id']][e['stage']] = e
    rows = []
    for trace_id, stages in groups.items():
        for name, start, end in EDGES:
            if start not in stages and end not in stages:
                continue
            row = dict(trace_id=trace_id, segment=name, ms=None, uncertainty_ms=None)
            if start not in stages or end not in stages:
                row['status'] = 'missing_event'
            elif (trace_id, start) in duplicate_keys or (trace_id, end) in duplicate_keys:
                row['status'] = 'ambiguous_duplicate'
            else:
                row['ms'], row['uncertainty_ms'], row['status'] = duration(stages[start], stages[end], sync or {})
            rows.append(row)
        # Exact round trip minus remote service interval; cannot split directions.
        keys = ['request_send_start', 'reply_received', 'server_received', 'reply_send_start']
        if all(k in stages for k in keys) and not any((trace_id,k) in duplicate_keys for k in keys):
            rtt, _, _ = duration(stages[keys[0]], stages[keys[1]], {})
            service, _, _ = duration(stages[keys[2]], stages[keys[3]], {})
            if rtt is not None and service is not None:
                residual = rtt-service
                rows.append(dict(trace_id=trace_id, segment='transport_and_serialization_roundtrip',
                                 ms=residual if residual >= 0 else None, uncertainty_ms=0,
                                 status='local_intervals' if residual >= 0 else 'invalid_order'))
    return rows


def collect(args):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('127.0.0.1', args.port))
    sock.settimeout(0.1)
    node = pub = rclpy = None
    if args.ros:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        from std_msgs.msg import String
        rclpy.init(args=[])
        node = Node('g1_latency_' + HOST.replace('-', '_').replace('.', '_'))
        pub = node.create_publisher(String, '/g1_latency/events',
                                   QoSProfile(depth=2048, reliability=ReliabilityPolicy.BEST_EFFORT))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    print('Recording local trace events:', args.output, flush=True)
    recent = collections.deque(maxlen=10000)
    last_summary = time.monotonic()
    try:
        with open(args.output, 'a', buffering=1) as f:
            while True:
                if time.monotonic() - last_summary >= args.summary_every:
                    print_summary(analyze(list(recent)))
                    last_summary = time.monotonic()
                try:
                    wire = sock.recv(65535)
                except socket.timeout:
                    if node:
                        rclpy.spin_once(node, timeout_sec=0)
                    continue
                try:
                    e = json.loads(wire)
                    if e.get('v') != 1 or not all(k in e for k in ('clock','trace_id','stage','event_id')):
                        continue
                    text = json.dumps(e, allow_nan=False)
                except (ValueError, TypeError):
                    continue
                f.write(text + '\n')
                recent.append(e)
                if pub:
                    pub.publish(String(data=text))
                    rclpy.spin_once(node, timeout_sec=0)
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()
        if node:
            node.destroy_node()
            rclpy.shutdown()


def print_summary(rows):
    values = collections.defaultdict(list)
    unavailable = collections.Counter()
    for r in rows:
        if r['ms'] is not None:
            values[r['segment']].append(r['ms'])
        else:
            unavailable[r['status']] += 1
    for name, vv in sorted(values.items()):
        vv.sort()
        print(f'{name}: n={len(vv)} mean={sum(vv)/len(vv):.2f}ms '
              f'p95={vv[max(0,math.ceil(.95*len(vv))-1)]:.2f}ms max={max(vv):.2f}ms', flush=True)
    if unavailable:
        print('Unavailable (never replaced with zero):', dict(unavailable), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='mode', required=True)
    c = sub.add_parser('collect')
    c.add_argument('--port', type=int, default=19555)
    c.add_argument('--output', required=True)
    c.add_argument('--ros', action='store_true', help='Publish telemetry to ROS 2, std_msgs/String')
    c.add_argument('--summary-every', type=float, default=5)
    f = sub.add_parser('relay', help='Relay JSONL from stdin (e.g. ssh tail -F) to local collector')
    f.add_argument('--port', type=int, default=19555)
    r = sub.add_parser('report')
    r.add_argument('files', nargs='+')
    r.add_argument('--sync', help='JSON clock calibration bounds covering recording period')
    r.add_argument('--output', default='latency_report.json')
    args = p.parse_args()
    if args.mode == 'relay':
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setblocking(False)
        for line in sys.stdin:
            try:
                e = json.loads(line)
                if e.get('v') == 1 and len(line.encode()) <= 8192:
                    sock.sendto(line.encode(), ('127.0.0.1', args.port))
            except (ValueError, OSError):
                print('Dropped malformed/unavailable relay event', file=sys.stderr)
        sock.close()
        return
    if args.mode == 'collect':
        collect(args)
        return
    events = []
    for path in args.files:
        with open(path) as f:
            for line in f:
                if line.strip():
                    events.append(json.loads(line))
    rows = analyze(events, json.loads(Path(args.sync).read_text()) if args.sync else None)
    summary = {}
    for name in sorted({r['segment'] for r in rows}):
        rr = [r for r in rows if r['segment']==name]
        values = sorted(r['ms'] for r in rr if r['ms'] is not None)
        summary[name] = dict(n=len(values), unavailable=len(rr)-len(values),
                             statuses=dict(collections.Counter(r['status'] for r in rr)))
        if values:
            summary[name].update(mean_ms=sum(values)/len(values),
                                 p95_ms=values[max(0, math.ceil(.95*len(values))-1)], max_ms=max(values),
                                 max_uncertainty_ms=max(r['uncertainty_ms'] or 0 for r in rr if r['ms'] is not None))
        print(name, json.dumps(summary[name]))
    Path(args.output).write_text(json.dumps(dict(summary=summary, samples=rows), indent=2))


if __name__ == '__main__':
    main()
