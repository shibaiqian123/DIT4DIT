from pathlib import Path
import difflib, hashlib, json
base=Path(__file__).parent
src=base/'source_snapshot'
manifest={}
patch=[]
for name in ['serve.py','model_adapter.py','model_adapter_32.py','run_dit4dit_rtc_sonic_dex1.py','run_pico_usb_stereo.py']:
    original=(src/name).read_text()
    text=original
    def replace(old,new):
        global text
        if text.count(old)!=1:
            raise RuntimeError((name,old[:100],text.count(old)))
        text=text.replace(old,new)
    # All optional diagnostics run through a nonthrowing SDK.
    replace('import argparse\n' if not name.startswith('model_adapter') else 'import numpy as np\n',
            ('import argparse\n' if not name.startswith('model_adapter') else 'import numpy as np\n')+'from latency_trace import mark, stamp\n')
    if name=='serve.py':
        replace("                r = await socket.receive_json()", "                r = await socket.receive_json()\n                mark(r.get('request_id', 'unbound'), 'server_received')")
        replace("                r['observation'] = obs", "                obs = dict(obs)\n                obs['_latency_request'] = r['request_id']\n                r['observation'] = obs")
        replace("                    await send(dict(result, ok=True", "                    mark(r['request_id'], 'reply_send_start')\n                    await send(dict(result, ok=True")
    elif name.startswith('model_adapter'):
        replace('        ex = self.example(obs)', "        trace_id = obs.get('_latency_request', obs['observation_id'])\n        mark(trace_id, 'predict_start')\n        ex = self.example(obs)")
        replace('        start = time.monotonic()', "        start = time.monotonic()\n        mark(trace_id, 'preprocess_done')")
        replace('        return dict(normalized=norm.tolist(),', "        mark(trace_id, 'predict_done')\n        return dict(normalized=norm.tolist(),")
    elif name=='run_pico_usb_stereo.py':
        replace('import argparse\n', 'import argparse\nimport uuid\n')
        replace('                        start=time.monotonic()', "                        camera_trace = uuid.uuid4().hex\n                        mark(camera_trace, 'camera_request_229')\n                        start=time.monotonic()")
        replace('                        data=read_exact(sock,size)', "                        data=read_exact(sock,size)\n                        mark(camera_trace, 'camera_payload_229', bytes=size)")
        replace('                        self.last_rpc_ms=', "                        mark(camera_trace, 'camera_jpeg_decoded_229')\n                        self.last_rpc_ms=")
    else:
        replace("                self.sensor_stage = 'camera_read'", "                obs_build_start = stamp()\n                self.sensor_stage = 'camera_read'")
        replace('                    with self.s.lock:\n                        self.s.observation, self.s.sources = obs, source_times',
                "                    obs['_latency_build_start'] = obs_build_start\n                    obs['_latency_ready'] = stamp()\n                    with self.s.lock:\n                        self.s.observation, self.s.sources = obs, source_times")
        replace('                    await ws.send(await asyncio.to_thread(json.dumps, packet))',
                "                    tid = packet.get('request_id')\n                    mark(tid, 'request_encode_start')\n                    wire = await asyncio.to_thread(json.dumps, packet)\n                    mark(tid, 'request_send_start')\n                    await ws.send(wire)\n                    mark(tid, 'request_send_done')")
        replace('                received_at = time.monotonic()', '                receive_stamp = stamp()\n                received_at = time.monotonic()')
        replace('                result = await asyncio.to_thread(json.loads, wire)',
                "                result = await asyncio.to_thread(json.loads, wire)\n                mark(result.get('request_id'), 'reply_received', at=receive_stamp)\n                mark(result.get('request_id'), 'reply_decoded')")
        replace("                req = clock.request(uuid.uuid4().hex); request_time = now",
                "                req = clock.request(uuid.uuid4().hex); request_time = now\n                tid = req['request_id']\n                for field, stage in [('_latency_build_start', 'obs_build_start'), ('_latency_ready', 'obs_ready')]:\n                    if field in requested:\n                        mark(tid, stage, at=requested[field], observation_id=requested['observation_id'])\n                mark(tid, 'request_queued', observation_id=requested['observation_id'])")
        replace('                        response = result; blocks += 1',
                "                        response = result; blocks += 1\n                        response['_latency_accepted'] = stamp()\n                        mark(result['request_id'], 'reply_accepted', at=response['_latency_accepted'])")
        replace('                        if a.execute:\n                            pub.send(',
                "                        if a.execute:\n                            action_trace = f'{clock.session}:frame:{frame}'\n                            mark(action_trace, 'action_ready', at=response['_latency_accepted'], request_id=response['request_id'])\n                            mark(action_trace, 'action_send', wire_frame_index=frame, request_id=response['request_id'])\n                            pub.send(")
    compile(text,name,'exec')
    manifest[name]={'before':hashlib.sha256(original.encode()).hexdigest(),'after':hashlib.sha256(text.encode()).hexdigest()}
    (base/'instrumented').mkdir(exist_ok=True)
    (base/'instrumented'/name).write_text(text)
    patch.extend(difflib.unified_diff(original.splitlines(True),text.splitlines(True),fromfile='a/'+name,tofile='b/'+name))
(base/'instrumentation.patch').write_text(''.join(patch))
(base/'manifest.json').write_text(json.dumps(manifest,indent=2))
