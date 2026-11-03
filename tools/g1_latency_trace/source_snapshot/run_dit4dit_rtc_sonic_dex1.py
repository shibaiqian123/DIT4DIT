"""Live DiT4DiT RTC -> SONIC v1.1 + physical Dex1, read-only by default.

Structure reference: Psi0 real/SONIC/run_psi0_rtc_sonic_dex1.py.
No virtual hand14 conversion or FSQ quantization: this policy uses raw state31
and denormalized token64 + Dex1 motor-radian2. No automatic SONIC start/stop.
Automatic network-loss stopping at the SONIC receiver is NOT closed-loop.
"""
import argparse
import asyncio
import base64
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import select
import socket
import sys
import threading
import time
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np
from protocol import ConsumptionClock, HZ

SIDES = ('left', 'right')
WARNING = 'Automatic network-loss stopping at SONIC is NOT closed-loop; use controller O / hardware emergency stop.'


def dependencies():
    # Imports only: --check never constructs a sensor, DDS channel or publisher.
    import websockets
    import zmq
    from PIL import Image
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from gear_sonic.camera.composed_camera import ComposedCameraClientSensor
    from gear_sonic.data.robot_model.instantiation.g1 import instantiate_g1_robot_model
    from gear_sonic.scripts.run_dex1_trigger_control import DdsGrippers, PacketReader, read_limits
    from gear_sonic.utils.inference.model_only_control import ModelGripInput
    from gear_sonic.utils.inference.observation_diagnostics import inspect_observation
    from gear_sonic.utils.inference.readonly_body import ReadOnlyBodySubscriber
    from gear_sonic.utils.inference.dit4dit_control import sonic_state, target_to_trigger
    from gear_sonic.utils.teleop.dex1_trigger import RampController
    from gear_sonic.utils.teleop.zmq.zmq_planner_sender import pack_pose_message
    from readonly_client import Recorder
    return locals()


def validate_info(info):
    if (info.get('state_dim') != 31 or info.get('action_shape') != [50, 66]
            or info.get('token_shape') != [50, 64] or info.get('control_hz') != 50
            or info.get('rtc_mode') != 'test_time' or info.get('rtc_supported') is not True):
        raise ValueError('Server is not the expected DiT4DiT state31/action50x66 RTC policy')
    for k in ('checkpoint_sha256', 'config_sha256', 'statistics_sha256'):
        v = info.get(k)
        if not isinstance(v, str) or len(v) != 64 or any(c not in '0123456789abcdef' for c in v):
            raise ValueError('Missing resource hash: ' + k)
    horizon = info.get('action_exec_horizon_default')
    if type(horizon) is not int or not 1 <= horizon < 50:
        raise ValueError('Invalid server action_exec_horizon')
    return horizon


def validate_reply(result, requested, info, limits):
    if result.get('ok') is not True:
        raise ValueError(result.get('error', 'Inference failed'))
    if any(result.get(k) != requested[k] for k in ('observation_id', 'observed_at')):
        raise ValueError('Reply observation provenance mismatch')
    resources = result.get('resources', {})
    for key in ('checkpoint_sha256', 'config_sha256', 'statistics_sha256',
                'state_dim', 'action_shape', 'token_shape', 'rtc_mode', 'inference_steps'):
        if resources.get(key) != info.get(key):
            raise ValueError('Server resources changed: ' + key)
    token = np.asarray(result['motion_tokens'], dtype=np.float32)
    grip = np.asarray(result['gripper_actions'], dtype=np.float64)
    if token.shape != (50, 64) or grip.shape != (50, 2) or not np.isfinite(token).all() or not np.isfinite(grip).all():
        raise ValueError('Invalid returned token/gripper block')
    for i, side in enumerate(SIDES):
        lim = limits[side]
        if np.any(grip[:, i] < min(lim.closed, lim.opened)) or np.any(grip[:, i] > max(lim.closed, lim.opened)):
            raise ValueError('Prediction outside calibrated Dex1 endpoints: ' + side)
    return token, grip


class Shared:
    """One fault latch also serializes token publication against watchdog faults."""
    def __init__(self):
        self.lock = threading.RLock()
        self.fault = None
        self.armed = False
        self.arm_id = uuid.uuid4().hex
        self.target = None
        self.heartbeat = None
        self.observation = None
        self.sources = None
        self.sensor_error = 'waiting for observation'
        self.grip_phase = 'WAIT_INPUTS'

    def fail(self, reason):
        with self.lock:
            if self.fault is None:
                self.fault = str(reason)
                self.target = None
                print('FAULT_LOCKED:', reason, '\n' + WARNING, flush=True)

    def bottom(self, now, states=('WAIT_FOR_CONTROL', 'CONTROL')):
        p = self.heartbeat or {}
        return (p.get('stop') is False and p.get('state') in states
                and isinstance(p.get('time'), (int, float)) and 0 <= now-p['time'] <= .25)

    def observation_now(self, now):
        if self.observation is None or self.sources is None:
            raise ValueError(self.sensor_error)
        if any(not 0 <= now-t <= .25 for t in self.sources):
            raise ValueError('Cached observation expired; source times are not renewed')
        if max(self.sources)-min(self.sources) > .15:
            raise ValueError('Observation skew exceeds 150ms')
        return dict(self.observation)


class Runtime:
    def __init__(self, a, deps, shared, limits, recorder):
        self.a, self.d, self.s, self.limits = a, deps, shared, limits
        self.recorder = recorder
        self.stop = threading.Event()
        self.dds = None
        self.threads = []
        self.sensor_stage = "not_started"
        self.gripper_stage = "not_started"

    def start(self):
        # URDF/model construction can hold the GIL for seconds. Complete it
        # before DDS and watchdog threads exist, so startup cannot queue stale
        # heartbeats or pause an already-running gripper watchdog.
        self.robot = self.d['instantiate_g1_robot_model'](waist_location='lower_and_upper_body')
        self.dds = self.d['DdsGrippers'](self.a.network, execute=self.a.execute)
        for fn in (self.heartbeats, self.sensors, self.grippers):
            t = threading.Thread(target=fn, daemon=True)
            self.threads.append(t)
            t.start()

    def heartbeats(self):
        # Keep status polling independent of image decode/resize/PNG encoding.
        ctx = sock = None
        try:
            z = self.d['zmq']; ctx = z.Context(); sock = ctx.socket(z.SUB)
            sock.setsockopt(z.LINGER, 0)
            sock.setsockopt(z.SUBSCRIBE, b'dex1_control_status ')
            sock.connect(f'tcp://127.0.0.1:{self.a.zmq_sub_port}')
            reader = self.d['PacketReader'](sock, b'dex1_control_status ')
            previous = None
            last_poll = time.monotonic()
            last_lock_wait = last_lock_hold = 0.0
            while not self.stop.is_set():
                poll_start = time.monotonic()
                poll_gap = poll_start - last_poll
                last_poll = poll_start
                hb = reader.poll()
                poll_end = time.monotonic()
                if poll_gap > .05 or poll_end-poll_start > .05 or (hb and hb.get('reader_error')):
                    self.recorder.add(None, dict(kind='heartbeat_timing', time=poll_end,
                        poll_gap_seconds=poll_gap, poll_seconds=poll_end-poll_start,
                        previous_lock_wait_seconds=last_lock_wait,
                        previous_lock_hold_seconds=last_lock_hold,
                        sensor_stage=self.sensor_stage, gripper_stage=self.gripper_stage,
                        reader_error=None if hb is None else hb.get('reader_error')))
                lock_start = time.monotonic()
                with self.s.lock:
                    lock_acquired = time.monotonic()
                    last_lock_wait = lock_acquired-lock_start
                    self.s.heartbeat = hb
                    signature = None if hb is None else (hb.get('state'), hb.get('stop'), hb.get('external_stream'), hb.get('reader_error'))
                    if signature != previous:
                        self.recorder.add(None, dict(kind='bottom_status', time=time.monotonic(), heartbeat=hb))
                        previous = signature
                    if hb and hb.get('reader_error'):
                        self.s.fail('SONIC heartbeat rejected: ' + hb['reader_error'])
                    elif self.s.armed and not self.s.bottom(time.monotonic()):
                        self.s.fail('SONIC stopped or heartbeat stale')
                last_lock_hold = time.monotonic()-lock_acquired
                self.stop.wait(.005)
        except Exception as exc:
            self.s.fail('Heartbeat worker: ' + str(exc))
        finally:
            if sock is not None: sock.close()
            if ctx is not None: ctx.term()

    def sensors(self):
        camera = body = None
        try:
            d, a = self.d, self.a
            robot = self.robot
            self.sensor_stage = 'camera_init'
            camera = d['ComposedCameraClientSensor']('127.0.0.1', a.camera_port)
            self.sensor_stage = 'body_init'
            body = d['ReadOnlyBodySubscriber'](a.body_port)
            while not self.stop.is_set():
                self.sensor_stage = 'camera_read'
                cm = camera.read()
                self.sensor_stage = 'body_read'
                bm = body.get_msg()
                self.sensor_stage = 'observation_validation'
                measured = self.dds.snapshot(); now = time.monotonic()
                dex = dict(time=now, sides={s: dict(q=m[0], feedback_time=m[1],
                           feedback_valid=bool(np.isfinite(m).all())) for s, m in measured.items()})
                diag = d['inspect_observation'](cm, bm, dex, self.limits, now, time.time(), body.cache.error)
                try:
                    if not diag['ok'] or bm is None or bm.get('valid') is not True:
                        raise ValueError('; '.join(diag['reasons']) or 'invalid body state')
                    q = np.asarray(bm['body_q'], dtype=float)
                    self.sensor_stage = 'state_conversion'
                    whole = robot.get_configuration_from_actuated_joints(body_actuated_joint_values=q)
                    state = d['sonic_state'](whole, [measured[s][0] for s in SIDES])
                    image_time = now-diag['camera_age_seconds']
                    source_times = [image_time, bm['sample_time']] + [measured[s][1] for s in SIDES]
                    self.sensor_stage = 'image_encode'
                    b = io.BytesIO()
                    d['Image'].fromarray(np.asarray(cm['images']['ego_view']).copy()).resize(
                        (224, 224), d['Image'].Resampling.BILINEAR).save(b, format='PNG')
                    obs = dict(image_png=base64.b64encode(b.getvalue()).decode(),
                               state=state.tolist(), lang=a.instruction, observation_id=uuid.uuid4().hex,
                               observed_at=image_time, saved_observation=False,
                               body_source_tick=bm['source_tick'], source_times=source_times)
                    with self.s.lock:
                        self.s.observation, self.s.sources = obs, source_times
                        self.s.sensor_error = ''
                except (ValueError, KeyError, TypeError) as exc:
                    with self.s.lock:
                        self.s.observation = None
                        self.s.sensor_error = str(exc)
                        if self.s.armed: self.s.fail(exc)
                self.sensor_stage = 'idle'
                self.stop.wait(.01)
        except Exception as exc:
            self.s.fail('Observation worker: ' + str(exc))
        finally:
            if camera is not None: camera.close()
            if body is not None: body.close()

    def grippers(self):
        d, s = self.d, self.s
        c = d['RampController'](self.limits, timeout=.25, obstruction_hold=True, obstruction_timeout=.5)
        model = d['ModelGripInput'](allow_default_reference_control=True)
        deadline = time.monotonic()
        try:
            while not self.stop.is_set():
                with s.lock:
                    measured = self.dds.snapshot()
                    now = time.monotonic()
                    if not s.armed:
                        packet = None
                    else:
                        packet = s.target
                        if packet is None or not 0 <= now-packet['time'] <= .25:
                            s.fail('Dex1 local action watchdog expired')
                        if not s.bottom(now): s.fail('SONIC heartbeat invalid')
                        if s.fault:
                            packet = dict(time=now, sample_time=0., valid=False, stop=True)
                    self.gripper_stage = 'model_update'
                    targets = model.update(c, packet, measured, s.heartbeat, now)
                    s.grip_phase = c.phase
                    if s.armed and c.phase == 'FAULT_HOLD': s.fail('Dex1: ' + c.reason)
                    # FAULT_HOLD is existing behavior, NOT a verified safe state.
                    self.gripper_stage = 'dds_write'
                    self.dds.write(targets)
                    self.gripper_stage = 'post_write'
                    if s.armed:
                        try:
                            self.recorder.add(None, dict(kind='dex1_cycle', time=now,
                                phase=c.phase, reason=c.reason, measured=measured,
                                requested=dict(c.requested), targets=dict(targets),
                                writes=dict(self.dds.last_writes), obstruction=c.obstruction_status,
                                motor_execution_confirmed=False))
                        except Exception as exc:
                            # Losing diagnostics must not terminate the hold/watchdog worker.
                            s.fail('Diagnostic writer failed: ' + str(exc))
                deadline += .02
                if time.monotonic()-deadline > .02:
                    if s.armed: s.fail('Dex1 scheduling deadline missed')
                    deadline = time.monotonic()
                self.stop.wait(max(0., deadline-time.monotonic()))
        except Exception as exc:
            s.fail('Dex1 writer failed; holding cannot be guaranteed: ' + str(exc))

    def close(self):
        self.stop.set()
        for t in self.threads: t.join(timeout=2)
        if any(t.is_alive() for t in self.threads):
            print('Worker did not stop promptly; process exit ends software holding.', flush=True)
        if self.dds is not None: self.dds.close()


def control_packet(s, mode, now, grip=None):
    p = dict(time=now, sample_time=now, valid=True, stop=False,
             mode=mode, arm_id=s.arm_id)
    if grip is not None: p.update(zip(SIDES, map(float, grip)))
    return p


def conflict_check(a):
    names = ('gear_sonic.scripts.run_dit4dit_model', 'gear_sonic.scripts.run_dex1_trigger_control',
             'run_dit4dit_model.py', 'run_dex1_trigger_control.py',
             'run_dit4dit_rtc_sonic_dex1.py', 'run_psi0_rtc_sonic_dex1.py',
             'psi_rtc_sonic_client.py', 'sonic_single_step')
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        if int(path.parent.name) == os.getpid(): continue
        try: args = path.read_bytes().split(b'\0')
        except (OSError, PermissionError): continue
        if not args or 'python' not in Path(os.fsdecode(args[0])).name: continue
        if any(any(name in os.fsdecode(arg) for name in names) for arg in args[1:]):
            raise RuntimeError('Conflicting robot client/Dex1 process PID ' + path.parent.name)
    for port in (a.zmq_pub_port, 5563, 5559):
        with socket.socket() as probe:
            probe.settimeout(.2)
            if probe.connect_ex(('127.0.0.1', port)) == 0:
                raise RuntimeError(f'Conflicting local publisher port {port}; no process was stopped')


class WarmupRetry(Exception):
    """Only an expired, unpublished first prediction may reset the RTC session."""


async def session(a, d, s, info, limits, recorder):
    for attempt in range(2):
        try:
            return await session_once(a, d, s, info, limits, recorder, attempt)
        except WarmupRetry:
            print('WARMUP_DISCARDED: new connection, new observation, no action history', flush=True)
            # Wait for a camera sample captured after warmup, not merely a new UUID.
            completed = time.monotonic()
            while True:
                with s.lock:
                    now = time.monotonic()
                    if s.fault: raise RuntimeError(s.fault)
                    if a.execute:
                        if not s.bottom(now, ('CONTROL',)) or (s.heartbeat or {}).get('external_stream') != 0:
                            s.fail('SONIC changed state during warmup; stop manually')
                            raise RuntimeError(s.fault)
                        s.target = control_packet(s, 'hold', now)
                    try: obs = s.observation_now(now)
                    except ValueError: obs = None
                    if obs is not None and obs['observed_at'] > completed: break
                    if now-completed > 10:
                        s.fail('No new observation after warmup')
                        raise TimeoutError(s.fault)
                await asyncio.sleep(.001)


async def session_once(a, d, s, info, limits, recorder, attempt):
    import websockets
    import zmq
    clock = ConsumptionClock(uuid.uuid4().hex, info['action_exec_horizon_default'])
    ctx = pub = None
    tasks = []
    warmup_retry = False
    if a.execute:
        ctx = zmq.Context(); pub = ctx.socket(zmq.PUB)
        pub.setsockopt(zmq.LINGER, 0); pub.setsockopt(zmq.SNDHWM, 1)
        pub.bind(f'tcp://127.0.0.1:{a.zmq_pub_port}')
    def emit(kind, **data):
        recorder.add(None, dict(kind=kind, time=time.monotonic(), **data))
    try:
        async with websockets.connect(f'ws://{a.host}:{a.port}/ws', compression=None,
                                      max_size=16*1024*1024, open_timeout=10) as ws:
            outgoing = asyncio.Queue(maxsize=1)
            async def sender():
                while True:
                    packet = await outgoing.get()
                    emit("send_started", request_id=packet.get("request_id"))
                    await ws.send(await asyncio.to_thread(json.dumps, packet))
                    emit("send_finished", request_id=packet.get("request_id"))
            async def receiver():
                wire = await ws.recv()
                received_at = time.monotonic()
                result = await asyncio.to_thread(json.loads, wire)
                emit("reply_received", received_at=received_at,
                     request_id=result.get("request_id"), ok=result.get("ok"),
                     inference_seconds=result.get("inference_seconds"))
                return result
            send = asyncio.create_task(sender()); receive = asyncio.create_task(receiver())
            tasks = [send, receive]
            requested = None; request_time = None; deadline = None
            response = tokens = grips = None
            running = not a.execute; arm_pending = False; frame = 0; blocks = 0
            stream_seen = False; last_receive = None; last_stream = None
            wait_start = time.monotonic()
            print('Commands: stop, quit. Waiting for default-reference CONTROL; no arm/run needed.', flush=True)
            def request(now):
                nonlocal requested, request_time
                requested = s.observation_now(now)
                req = clock.request(uuid.uuid4().hex); request_time = now
                emit('request', **req, observation_id=requested['observation_id'],
                     observed_at=requested['observed_at'], source_times=requested['source_times'])
                recorder.add('request_' + req['request_id'] + '.json', dict(req, observation=requested))
                outgoing.put_nowait(dict(req, op='infer', observation=requested))
            while True:
                now = time.monotonic()
                if send.done(): send.result()
                if a.execute and select.select([sys.stdin], [], [], 0)[0]:
                    line = sys.stdin.readline()
                    if not line: raise RuntimeError('Control terminal closed')
                    key = line.strip().lower()
                    with s.lock:
                        now = time.monotonic()
                        if key == 'quit': return
                        if key == 'stop': raise RuntimeError('Manual stop')
                        if key: print('Commands: stop, quit. Enable stream manually in SONIC.', flush=True)
                with s.lock:
                    now = time.monotonic()
                    if s.fault: raise RuntimeError(s.fault)
                    stream = None
                    if a.execute:
                        hb = s.heartbeat or {}
                        if hb.get('reader_error'):
                            raise RuntimeError('SONIC heartbeat rejected: ' + hb['reader_error'])
                        if hb.get('stop') is True:
                            raise RuntimeError('SONIC reports STOP; no model enable')
                        stream = hb.get('external_stream')
                        if hb and (type(stream) is not int or stream not in (0, 1)):
                            raise RuntimeError('Missing/unknown external_stream; rebuild SONIC before enabling')
                        if s.bottom(now, ('CONTROL',)) and stream == 1 and frame == 0:
                            raise RuntimeError('External stream enabled before first publication; stop SONIC manually')
                        if not s.armed:
                            if not s.bottom(now, ('CONTROL',)) or stream != 0:
                                await asyncio.sleep(.001)
                                continue
                            try: s.observation_now(now)
                            except ValueError:
                                await asyncio.sleep(.001)
                                continue
                            s.armed = True; arm_pending = True
                            emit('automatic_hold_initialization')
                        s.observation_now(now)
                        if not s.bottom(now, ('CONTROL',)):
                            raise RuntimeError('Bottom stopped/stale/not CONTROL')
                        if stream != last_stream:
                            emit('stream_state_observed', external_stream=stream,
                                 heartbeat_time=hb['time'], heartbeat_age=now-hb['time'],
                                 hardware_synchronization_confirmed=False)
                            print('EXTERNAL_STREAM:', stream, flush=True)
                            last_stream = stream
                        if stream_seen and stream == 0:
                            raise RuntimeError('External stream disabled; delivery stopped, restart client to resume')
                        stream_seen = stream_seen or stream == 1
                        if arm_pending and s.grip_phase == 'READY':
                            arm_pending = False
                            print('GRIPPERS_READY: holding current opening', flush=True)
                        if s.grip_phase == 'READY': running = True
                        if deadline is None or stream == 0:
                            s.target = control_packet(s, 'arm' if arm_pending else 'hold', now)
                    if running and requested is None:
                        try: request(now)
                        except ValueError:
                            if a.execute or now-wait_start > 10: raise
                    if receive.done():
                        result = receive.result()
                        if requested is None: raise ValueError('Unsolicited server reply')
                        tokens, grips = validate_reply(result, requested, info, limits)
                        age = now-requested['observed_at']
                        if not 0 <= age < (0.5 if deadline is None else 1.0):
                            if age >= .5 and deadline is None and frame == 0 and attempt == 0 and (not a.execute or stream == 0):
                                recorder.add('warmup_discarded.json', dict(result=result, observation_age=age))
                                raise WarmupRetry()
                            raise ValueError('Prediction too old for execution/consumption')
                        if not clock.accept(result): raise ValueError('Old-session result')
                        response = result; blocks += 1
                        recorder.add(f'chunk_{blocks:04d}.json', result)
                        interval = None if last_receive is None else now-last_receive
                        print(f'Received action: chunk={clock.chunk_id} age={age:.3f}s external_stream={stream}; recv_action interval={interval}', flush=True)
                        last_receive = now
                        emit('accepted', chunk_id=clock.chunk_id, skipped_prefix=clock.position, observation_age=age)
                        if deadline is None: deadline = now
                        receive = asyncio.create_task(receiver()); tasks = [send, receive]
                    if clock.pending is not None and now-request_time > 10:
                        raise TimeoutError('Inference timeout; no automatic retry')
                    if deadline is not None and now >= deadline:
                        if now-deadline >= 1/HZ: raise RuntimeError('Missed whole control tick; no burst catch-up')
                        s.observation_now(now)
                        if not 0 <= now-response['observed_at'] < 1.0: raise RuntimeError('Action block observation expired')
                        if a.execute and not s.bottom(now, ('CONTROL',)): raise RuntimeError('Bottom not CONTROL')
                        if a.execute and s.grip_phase != 'READY': raise RuntimeError('Dex1 not READY')
                        index = clock.consume()
                        packet = control_packet(s, 'model', now, grips[index])
                        packet = d['target_to_trigger'](packet, limits, now)
                        if a.execute:
                            pub.send(d['pack_pose_message']({'token_state': tokens[index].reshape(1,64),
                                'frame_index': np.array([frame], dtype=np.int64)}, topic='pose', version=4), flags=zmq.NOBLOCK)
                            if stream == 1: s.target = packet
                            if frame == 0:
                                print('TOKEN_PUBLISHING: valid first block published. Press Enter in SONIC to enable stream.', flush=True)
                        emit('delivered' if a.execute else 'readonly_consumed', frame_index=frame,
                             chunk_id=clock.chunk_id, index=index, token=tokens[index].tolist(),
                             gripper_raw=grips[index].tolist(), gripper_applied=bool(a.execute and stream == 1),
                             external_stream=stream, schedule_lateness=now-deadline,
                             motor_execution_confirmed=False)
                        frame += 1; deadline += 1/HZ
                        if clock.ready_to_request(): request(now)
                await asyncio.sleep(.001)
    except WarmupRetry:
        warmup_retry = True
        raise
    except BaseException as exc:
        s.fail(str(exc) or type(exc).__name__)
        raise
    finally:
        # Latch before closing the connection and before any awaited cleanup.
        if not warmup_retry: s.fail('Client session ended; no automatic resume')
        clock.failed = True; clock.pending = None
        for task in tasks: task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if pub is not None: pub.close()
        if ctx is not None: ctx.term()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8014)
    p.add_argument('--network', default='enx6c1ff7b8d821')
    p.add_argument('--instruction')
    p.add_argument('--camera-port', type=int, default=5555)
    p.add_argument('--body-port', type=int, default=5564)
    p.add_argument('--zmq-pub-port', type=int, default=5556)
    p.add_argument('--zmq-sub-port', type=int, default=5557)
    p.add_argument('--limits', type=Path, default=ROOT/'gear_sonic/configs/dex1_limits.g1_20261013.json')
    p.add_argument('--output', type=Path)
    p.add_argument('--check', action='store_true', help='Imports/calibration only; no sensors, sockets or DDS')
    p.add_argument('--execute', action='store_true', help='Enable publishers after default-reference CONTROL; SONIC stream switch remains manual')
    a = p.parse_args(argv)
    if a.check and a.execute: p.error('--check excludes --execute')
    if not a.check and (not a.instruction or not a.instruction.strip() or a.output is None):
        p.error('--instruction and a new --output directory are required')
    for port in (a.port, a.camera_port, a.body_port, a.zmq_pub_port, a.zmq_sub_port):
        if not 1024 <= port <= 65535: p.error('Ports must be 1024..65535')
    if a.execute and not sys.stdin.isatty(): p.error('--execute requires an interactive control terminal')
    return a


def main():
    a = parse_args(); d = dependencies(); limits = d['read_limits'](a.limits)
    if a.check:
        print('CLIENT_CHECK_OK: imports and calibrated limits; NO DDS/SENSORS/PUBLISHERS')
        return
    socket.if_nametoindex(a.network)
    conflict_check(a)
    # Cooperative lock prevents two instances racing through the process check.
    lock = open(f'/tmp/dit4dit-rtc-dex1-{os.getuid()}.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with urllib.request.urlopen(f'http://{a.host}:{a.port}/info', timeout=10) as f:
        info = json.load(f)
    validate_info(info)
    a.output.mkdir(parents=True, exist_ok=False)
    recorder = d['Recorder'](a.output)
    recorder.add('session.json', dict(server=info, execute=a.execute,
        limits_sha256=hashlib.sha256(a.limits.read_bytes()).hexdigest(), instruction=a.instruction,
        receiver_network_stop_closed_loop=False, gripper_hold_hardware_validated=False))
    s = Shared(); runtime = Runtime(a, d, s, limits, recorder)
    print(WARNING, '\nDex1 FAULT_HOLD reuses existing logic; hardware safety NOT validated.', flush=True)
    try:
        runtime.start()
        try:
            asyncio.run(session(a, d, s, info, limits, recorder))
        except KeyboardInterrupt:
            s.fail('Keyboard interrupt')
        except Exception as exc:
            s.fail(exc)
            try:
                recorder.add(None, dict(kind='fault_locked', time=time.monotonic(), error=str(exc)))
            except Exception as log_error:
                print('Fault could not be recorded:', log_error, flush=True)
            if a.execute:
                print('No further token/action delivery. Dex1 fault worker remains active.\n'
                      'Stop SONIC at its terminal, then Ctrl+C here to release resources.', flush=True)
                try:
                    while True: time.sleep(.2)
                except KeyboardInterrupt: pass
            else: raise
    finally:
        s.fail('Closing client')
        runtime.close()
        recorder.close()
        lock.close()
        print('CLIENT_CLOSED: Dex1 software holding has ended. SONIC is NOT automatically stopped.', flush=True)


if __name__ == '__main__':
    main()
