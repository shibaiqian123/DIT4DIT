"""Loopback inference only. One owner, one GPU request, no robot publishers."""
import argparse
from latency_trace import mark, stamp
import asyncio
import json
import uuid
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
from protocol import shifted, block


def create_app(policy, action_exec_horizon=25, *, enable_preview_source=False, synchronous=False):
    if type(action_exec_horizon) is not int or not (1 <= action_exec_horizon <= 50 if synchronous else 1 <= action_exec_horizon < 50):
        raise ValueError("action_exec_horizon must be 1..50 in sync mode, otherwise 1..49")
    mode = policy.info.get('rtc_mode', 'test_time')
    if mode not in ('test_time', 'disabled'):
        raise ValueError('Unknown policy RTC mode')
    rtc_enabled = mode == 'test_time'
    if synchronous and rtc_enabled:
        raise ValueError('Synchronous execution requires --no-rtc')
    execution_mode = 'sync' if synchronous else 'async'
    resources = dict(policy.info, execution_mode=execution_mode)
    app = FastAPI()
    gpu_lock = asyncio.Lock()
    owner = None
    preview_latest = None

    @app.get('/health')
    async def health():
        return dict(status='ok', motor_interfaces=False)

    @app.get('/info')
    async def info():
        return dict(resources, action_exec_horizon_default=action_exec_horizon, control_hz=50,
                    timing='229_client_consumption_not_motor_ack', rtc_supported=True, rtc_enabled=rtc_enabled,
                    image_encodings=['png', 'rgb8_base64'], reply_formats=['full', 'actions_only_v1'],
                    preview_source_enabled=enable_preview_source)

    @app.get('/preview-observation')
    async def preview_observation(after: str = ''):
        # Pull-only diagnostic snapshot; action inference never waits for a viewer.
        if not enable_preview_source:
            return dict(enabled=False, active=False)
        if owner is None or preview_latest is None:
            return dict(enabled=True, active=False)
        age = time.monotonic()-preview_latest['received_mono']
        if age > 5:
            return dict(enabled=True, active=False, source_age_seconds=age)
        result = dict(enabled=True, active=True, key=preview_latest['key'],
                      source_age_seconds=age, resources=policy.info)
        if after != preview_latest['key']:
            result.update(observation=preview_latest['observation'],
                          request_id=preview_latest['request_id'], session=preview_latest['session'])
        return result

    @app.websocket('/ws')
    async def ws(socket: WebSocket):
        nonlocal owner, preview_latest
        await socket.accept()
        if owner is not None:
            await socket.close(code=1013, reason='An inference client is already connected')
            return
        owner = socket
        preview_latest = None
        pending = asyncio.Queue(maxsize=1)
        busy = False
        latest_obs = None
        session = None
        previous = None
        chunk_id = None
        last_consumed = 0
        seen = set()
        send_lock = asyncio.Lock()
        async def send(data):
            async with send_lock:
                await socket.send_json(data)
        async def reader():
            nonlocal latest_obs, busy, session
            while True:
                wire = await socket.receive_text()
                receive_stamp = stamp()
                r = json.loads(wire)
                mark(r.get('request_id', 'unbound'), 'server_received', at=receive_stamp)
                mark(r.get('request_id', 'unbound'), 'server_decoded')
                if r.get('op') == 'observe':
                    latest_obs = r['observation']  # single slot; never queues stale images
                    continue
                if r.get('op') != 'infer':
                    raise ValueError('Expected observe or infer')
                if busy or r.get('request_id') in seen:
                    raise ValueError('Duplicate or concurrent inference request')
                if not isinstance(r.get('session'), str) or not isinstance(r.get('request_id'), str):
                    raise ValueError('Missing request/session identity')
                if session is None:
                    session = r['session']
                if r['session'] != session:
                    raise ValueError('Session changed within connection')
                obs = r.get('observation', latest_obs)
                if obs is None:
                    raise ValueError('Observation missing')
                # Freeze observation and progress together; never substitute a later pose.
                obs = dict(obs)
                obs['_latency_request'] = r['request_id']
                r['observation'] = obs
                seen.add(r['request_id'])
                busy = True
                pending.put_nowait(r)
        async def worker():
            nonlocal previous, chunk_id, busy, last_consumed, preview_latest
            while True:
                r = await pending.get()
                obs = r['observation']
                try:
                    if r.get('execution_mode', 'async') != execution_mode:
                        raise ValueError('Client/server execution mode mismatch')
                    s, d = r['consumed'], r['delay']
                    if type(s) is not int or type(d) is not int or not (0 <= s <= (50 if synchronous else 49) and 0 <= d < 50):
                        raise ValueError('Invalid consumption/delay frames')
                    if synchronous and (d != 0 or s != (0 if chunk_id is None else action_exec_horizon)):
                        raise ValueError('Sync requests require exactly the configured execution steps and zero delay')
                    if r.get('chunk_id') != chunk_id:
                        raise ValueError('Unknown previous chunk')
                    if chunk_id is None:
                        if s != 0:
                            raise ValueError('First request consumed must be zero')
                        prefix = None
                    else:
                        if s < action_exec_horizon:
                            raise ValueError('Replan before configured action_exec_horizon')
                        if s < last_consumed:
                            raise ValueError('Consumption regressed')
                        prefix = shifted(previous, s, d) if rtc_enabled else None
                    if enable_preview_source:
                        preview_latest = dict(key=session+':'+r['request_id'], session=session,
                            request_id=r['request_id'], observation=obs, received_mono=time.monotonic())
                    async with gpu_lock:
                        task = asyncio.create_task(asyncio.to_thread(policy.predict, obs, prefix, d, s))
                        try:
                            result = await asyncio.shield(task)
                        except asyncio.CancelledError:
                            # Keep ownership until the uncancellable GPU call finishes.
                            await task
                            raise
                    import numpy as np
                    validated = block(result['normalized'])
                    previous = validated.copy() if rtc_enabled else None
                    reply_format = r.get('reply_format', 'full')
                    if reply_format not in ('full', 'actions_only_v1'):
                        raise ValueError('Unsupported reply format')
                    if reply_format == 'actions_only_v1':
                        # RTC history stays server-side, with its finite/shape guard.
                        result = {k:v for k,v in result.items() if k != 'normalized'}
                        result.update(reply_format=reply_format, normalized_validated=True)
                    chunk_id = uuid.uuid4().hex
                    last_consumed = 0
                    mark(r['request_id'], 'reply_send_start')
                    await send(dict(result, ok=True, session=session, request_id=r['request_id'],
                                    chunk_id=chunk_id, observation_id=obs['observation_id'],
                                    observed_at=obs['observed_at'], window_offset=s, estimated_delay=d,
                                    resources=resources))
                    busy = False
                except Exception as e:
                    await send(dict(ok=False, session=session, request_id=r['request_id'], error=str(e)))
                    raise
        tasks = [asyncio.create_task(reader()), asyncio.create_task(worker())]
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        except (WebSocketDisconnect, RuntimeError):
            pass
        except Exception as e:
            try:
                await send(dict(ok=False, session=session, error=str(e)))
            except Exception:
                pass
        finally:
            for task in tasks:
                task.cancel()
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
            finally:
                owner = None
            try:
                await socket.close()
            except Exception:
                pass
    return app


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', '--run-dir', dest='run', required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--checkpoint')
    g.add_argument('--ckpt-step', type=int)
    p.add_argument('--cosmos', required=True)
    p.add_argument('--port', type=int, default=8000)
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--action_exec_horizon', '--action-exec-horizon', type=int, default=25)
    p.add_argument('--policy', choices=['dit4dit'], default='dit4dit')
    p.add_argument('--enable-preview-source', action='store_true',
                   help='Expose latest inference input to a separate loopback preview GPU service')
    modes = p.add_mutually_exclusive_group()
    modes.add_argument('--rtc', dest='rtc_enabled', action='store_true', help='Enable test-time RTC (default)')
    modes.add_argument('--no-rtc', dest='rtc_enabled', action='store_false', help='Ordinary sampling for every chunk; keep async delivery')
    p.add_argument('--sync', action='store_true', help='Execute the selected prefix before the next inference; implies --no-rtc')
    p.add_argument('--sync-steps', type=int, default=None, metavar='N',
                   help='Rows to execute per prediction in --sync mode (1..50, default 50); model still predicts 50 rows')
    p.set_defaults(rtc_enabled=None)
    a = p.parse_args(argv)
    if a.sync_steps is not None and (not a.sync or not 1 <= a.sync_steps <= 50):
        p.error('--sync-steps requires --sync and must be 1..50')
    if a.sync:
        if a.rtc_enabled is True:
            p.error('--sync cannot be combined with --rtc')
        a.rtc_enabled = False
        a.action_exec_horizon = 50 if a.sync_steps is None else a.sync_steps
    else:
        if a.rtc_enabled is None: a.rtc_enabled = True
        if not 1 <= a.action_exec_horizon < 50:
            p.error('action_exec_horizon must be 1..49')
    if a.ckpt_step is not None:
        if a.ckpt_step <= 0: p.error('ckpt-step must be positive')
        a.checkpoint = f'steps_{a.ckpt_step}_pytorch_model.pt'
    return a


def main():
    a = parse_args()
    import socket
    with socket.socket() as probe:
        probe.bind((a.host, a.port))
    from model_adapter import Policy
    policy = Policy(a.run, a.checkpoint, a.cosmos, rtc_enabled=a.rtc_enabled)
    print('RTC_READY '+json.dumps(policy.info), flush=True)
    uvicorn.run(create_app(policy, a.action_exec_horizon, enable_preview_source=a.enable_preview_source, synchronous=a.sync), host=a.host, port=a.port, ws_max_size=16*1024*1024)

if __name__ == '__main__':
    main()
