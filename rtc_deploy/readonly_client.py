"""Replay saved observations, consume into a log ONLY. Never imports robot control."""
import argparse
import asyncio
import base64
import io
import json
import time
import uuid
import queue
import threading
from pathlib import Path
import numpy as np
from PIL import Image
import websockets
from protocol import ConsumptionClock, H, HZ


def read_observation(capture):
    manifest = json.loads((capture/'manifest.json').read_text())
    import hashlib
    for name, digest in manifest['files'].items():
        if hashlib.sha256((capture/name).read_bytes()).hexdigest() != digest:
            raise ValueError('Capture hash mismatch: '+name)
    with np.load(capture/'observation.npz', allow_pickle=False) as data:
        image, state = data['image'], data['state']
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError('Invalid saved RGB image')
        if state.shape != (31,) or not np.isfinite(state).all():
            raise ValueError('Invalid saved state')
        observed = float(data['observed_at'])
    b = io.BytesIO()
    Image.fromarray(image).resize((224,224), Image.Resampling.BILINEAR).save(b, format='PNG')
    return dict(image_png=base64.b64encode(b.getvalue()).decode(), state=state.tolist(),
                lang=manifest['task_prompt'], observed_at=observed,
                observation_id=uuid.uuid4().hex, saved_observation=True)


class Recorder:
    """Bounded disk worker: no file write/JSON serialization in the consumer tick."""
    def __init__(self, output):
        self.output=output;self.q=queue.Queue(maxsize=4096);self.error=None
        self.thread=threading.Thread(target=self.work,daemon=True);self.thread.start()
    def work(self):
        try:
            with (self.output/'events.jsonl').open('w') as log:
                while True:
                    item=self.q.get()
                    if item is None:return
                    name,data=item
                    if name is None:
                        log.write(json.dumps(data)+'\n');log.flush()
                    else:(self.output/name).write_text(json.dumps(data))
        except BaseException as e:self.error=e
    def add(self,name,data):
        if self.error:raise RuntimeError('Diagnostic writer failed') from self.error
        self.q.put_nowait((name,data))
    def close(self):
        while self.thread.is_alive():
            if self.error:break
            try:
                self.q.put(None,timeout=1)
                break
            except queue.Full:continue
        self.thread.join()
        if self.error:raise RuntimeError('Diagnostic writer failed') from self.error


async def run(a):
    observation = read_observation(a.capture)
    clock = ConsumptionClock(uuid.uuid4().hex, a.action_exec_horizon)
    a.output.mkdir(parents=True, exist_ok=False)
    recorder=Recorder(a.output)
    def emit(kind, **fields):
        recorder.add(None,dict(kind=kind,time=time.monotonic(),**fields))
    try:
        async with websockets.connect(a.url, max_size=16*1024*1024, compression=None) as ws:
            outgoing=asyncio.Queue(maxsize=1)
            async def sender():
                while True:
                    packet=await outgoing.get()
                    wire=await asyncio.to_thread(json.dumps,packet)
                    await ws.send(wire)
            async def get_reply():
                return await asyncio.to_thread(json.loads,await ws.recv())
            send_task=asyncio.create_task(sender())
            receive=asyncio.create_task(get_reply())
            requested_at=None;response=None;deadline=None;blocks=0
            def request():
                nonlocal requested_at
                req=clock.request(uuid.uuid4().hex)
                requested_at=time.monotonic()
                emit('request',**req)
                outgoing.put_nowait(dict(req,op='infer',observation=observation))
            request()
            try:
                while True:
                    if send_task.done():send_task.result()
                    if receive.done():
                        result=receive.result()
                        if result.get('ok') and (result.get('observation_id') != observation['observation_id'] or result.get('observed_at') != observation['observed_at']):
                            raise ValueError('Reply observation provenance mismatch')
                        old_chunk=clock.chunk_id
                        if not clock.accept(result):raise ValueError('Stale session reply')
                        for key,shape in [('motion_tokens',(50,64)),('gripper_actions',(50,2))]:
                            v=np.asarray(result[key])
                            if v.shape!=shape or not np.isfinite(v).all():raise ValueError('Invalid '+key)
                        response=result;blocks+=1
                        recorder.add(f'chunk_{blocks:04d}.json',result)
                        emit('accepted',chunk_id=clock.chunk_id,skipped_prefix=clock.position,
                             previous_chunk=old_chunk,client_delay_frames=clock.position)
                        if deadline is None:deadline=time.monotonic()
                        receive=asyncio.create_task(get_reply())
                    now=time.monotonic()
                    if clock.pending is not None and now-requested_at>10:
                        raise TimeoutError('Inference reply exceeded 10 seconds; no retry')
                    if deadline is not None and now>=deadline:
                        late=now-deadline
                        if late>=1/HZ:raise RuntimeError('Readonly scheduling missed a whole tick; no catch-up burst')
                        index=clock.consume()
                        emit('readonly_consumed',chunk_id=clock.chunk_id,index=index,
                             schedule_lateness=late,motor_send=False,
                             motion_token=response['motion_tokens'][index],
                             gripper_action=response['gripper_actions'][index])
                        deadline+=1/HZ
                        if clock.position>=a.action_exec_horizon and clock.pending is None:
                            if blocks>=a.blocks:
                                emit('complete',blocks=blocks,motor_send=False)
                                break
                            request()
                    await asyncio.sleep(.001)
            except Exception as e:
                emit('failed',error=str(e),motor_send=False)
                raise
            finally:
                receive.cancel();send_task.cancel()
                await asyncio.gather(receive,send_task,return_exceptions=True)
    finally:
        await asyncio.to_thread(recorder.close)
    print('READONLY_RTC_COMPLETE; NO MOTOR SEND',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--url', default='ws://127.0.0.1:8000/ws')
    p.add_argument('--blocks', type=int, default=3)
    p.add_argument('--action-exec-horizon', type=int, default=25)
    a=p.parse_args()
    if a.blocks < 1:
        p.error('blocks must be positive')
    asyncio.run(run(a))
if __name__ == '__main__':main()
