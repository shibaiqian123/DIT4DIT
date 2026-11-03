"""Loopback inference only. One owner, one GPU request, no robot publishers."""
import argparse
import asyncio
import json
import uuid
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
from protocol import shifted


def create_app(policy, action_exec_horizon=25):
    if not 1 <= action_exec_horizon < 50:
        raise ValueError("action_exec_horizon must be 1..49")
    app = FastAPI()
    gpu_lock = asyncio.Lock()
    owner = None

    @app.get('/health')
    async def health():
        return dict(status='ok', motor_interfaces=False)

    @app.get('/info')
    async def info():
        return dict(policy.info, action_exec_horizon_default=action_exec_horizon, control_hz=50,
                    timing='229_client_consumption_not_motor_ack', rtc_supported=True)

    @app.websocket('/ws')
    async def ws(socket: WebSocket):
        nonlocal owner
        await socket.accept()
        if owner is not None:
            await socket.close(code=1013, reason='An inference client is already connected')
            return
        owner = socket
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
                r = await socket.receive_json()
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
                r['observation'] = obs
                seen.add(r['request_id'])
                busy = True
                pending.put_nowait(r)
        async def worker():
            nonlocal previous, chunk_id, busy, last_consumed
            while True:
                r = await pending.get()
                obs = r['observation']
                try:
                    s, d = r['consumed'], r['delay']
                    if r.get('chunk_id') != chunk_id:
                        raise ValueError('Unknown previous chunk')
                    if previous is None:
                        if s != 0:
                            raise ValueError('First request consumed must be zero')
                        prefix = None
                    else:
                        if s < action_exec_horizon:
                            raise ValueError('Replan before configured action_exec_horizon')
                        if s < last_consumed:
                            raise ValueError('Consumption regressed')
                        prefix = shifted(previous, s, d)
                    async with gpu_lock:
                        task = asyncio.create_task(asyncio.to_thread(policy.predict, obs, prefix, d, s))
                        try:
                            result = await asyncio.shield(task)
                        except asyncio.CancelledError:
                            # Keep ownership until the uncancellable GPU call finishes.
                            await task
                            raise
                    import numpy as np
                    previous = np.asarray(result['normalized'], dtype=np.float32)
                    chunk_id = uuid.uuid4().hex
                    last_consumed = 0
                    await send(dict(result, ok=True, session=session, request_id=r['request_id'],
                                    chunk_id=chunk_id, observation_id=obs['observation_id'],
                                    observed_at=obs['observed_at'], window_offset=s, estimated_delay=d,
                                    resources=policy.info))
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
    p.add_argument('--rtc', action='store_true', help='Explicit test-time RTC selection (also the default)')
    a = p.parse_args(argv)
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
    policy = Policy(a.run, a.checkpoint, a.cosmos)
    print('RTC_READY '+json.dumps(policy.info), flush=True)
    uvicorn.run(create_app(policy, a.action_exec_horizon), host=a.host, port=a.port, ws_max_size=16*1024*1024)

if __name__ == '__main__':
    main()
