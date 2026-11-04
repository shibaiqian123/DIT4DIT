"""Independent-GPU, best-effort input/video preview. No motor/action publishers."""
import argparse
import asyncio
import base64
from contextlib import asynccontextmanager
import io
import json
import math
import os
import re
import time
import urllib.parse
import urllib.request
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn

IDENTITY = ('checkpoint_sha256', 'config_sha256', 'statistics_sha256')
UUID = re.compile(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}')


def check_source(info, preview_info, gpu_uuid):
    if not info.get('preview_source_enabled'):
        raise ValueError('RTC server needs --enable-preview-source')
    source_gpu = info.get('cuda_visible_devices', '')
    if not UUID.fullmatch(source_gpu) or not UUID.fullmatch(gpu_uuid):
        raise ValueError('Both services must select one GPU by its full GPU UUID')
    if source_gpu.lower() == gpu_uuid.lower():
        raise ValueError('Preview GPU must differ from RTC GPU')
    if preview_info is not None:
        for key in IDENTITY:
            if not info.get(key) or info[key] != preview_info.get(key):
                raise ValueError('Preview/RTC resource mismatch: '+key)


def fetch(url):
    # Loopback HTTP must not inherit an external HTTP proxy.
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(url, timeout=2) as r:
        return json.load(r)


def encode_preview(policy, observation):
    import numpy as np
    import torch
    from PIL import Image
    ex = policy.example(observation)
    backbone = policy.model.backbone_interface
    start = time.monotonic()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        inputs = backbone.build_cosmos_inputs(images=[ex['image']], instructions=[ex['lang']])
        video = backbone(**inputs, predict_future=True).pred_future_video
    if video is None or video.ndim != 5 or video.shape[0] != 1 or video.shape[2] != 3:
        raise ValueError('Expected generated video [1,T,3,H,W]')
    if not 1 <= video.shape[1] <= 256 or not bool(torch.isfinite(video).all()):
        raise ValueError('Invalid preview video length/values')
    # Decode is already [0,1]; no change to model inputs or robot commands.
    pixels = (video[0].detach().float().clamp(0,1)*255).round().byte().permute(0,2,3,1).cpu().numpy()
    def jpeg(image):
        b = io.BytesIO()
        image.convert('RGB').resize((224,224)).save(b,format='JPEG',quality=85)
        return 'data:image/jpeg;base64,'+base64.b64encode(b.getvalue()).decode()
    return dict(input_image=jpeg(ex['image'][0]), frames=[jpeg(Image.fromarray(f)) for f in pixels],
                generation_seconds=time.monotonic()-start, instruction=ex['lang'],
                observation_id=observation['observation_id'], observed_at=observation['observed_at'],
                note='Independent video prediction; not an RTC action rollout or motor acknowledgement')


PAGE = r'''<!doctype html><html lang="zh"><meta charset="utf-8">
<title>DiT4DiT 输入 / 预测预览</title><style>
body{background:#111827;color:#e5e7eb;font:16px system-ui;margin:30px}main{display:flex;gap:24px;flex-wrap:wrap}
section{flex:1;min-width:280px}img{width:100%;max-width:560px;image-rendering:auto;background:#000}
#status{color:#fbbf24}pre{white-space:pre-wrap}button{padding:8px 16px}</style>
<h1>模型输入与未来视频</h1><p id="status">连接预览服务…</p>
<p>两侧属于同一次输入。左侧保持该次输入帧，右侧循环播放预测片段；不是实时相机与真机动作的严格同步。</p>
<main><section><h2>对应的模型输入</h2><img id="input" alt="等待输入"></section>
<section><h2>独立GPU预测（包含条件帧）</h2><img id="prediction" alt="等待预测"></section></main>
<p><button id="pause">暂停播放</button> <span id="frame"></span></p><pre id="meta"></pre>
<script>
let key='',frames=[],index=0,playing=true,timer=null;
const el=id=>document.getElementById(id);
el('pause').onclick=()=>{playing=!playing;el('pause').textContent=playing?'暂停播放':'继续播放'};
async function poll(){try{
 const r=await fetch('/api/preview?after='+encodeURIComponent(key),{cache:'no-store'});if(!r.ok)throw Error(r.status);
 const d=await r.json();el('status').textContent=d.status+' '+(d.error||'')+(d.result_age_seconds==null?'':' | 结果已生成 '+d.result_age_seconds.toFixed(1)+' 秒');
 if(d.result){const v=d.result;key=v.key;frames=v.frames;index=0;el('input').src=v.input_image;el('prediction').src=frames[0];
 el('meta').textContent='观测: '+v.observation_id+'\n请求: '+v.key+'\n生成耗时: '+v.generation_seconds.toFixed(3)+' 秒\n源请求到预览抓取间隔: '+v.source_age_seconds.toFixed(3)+' 秒\n播放速率: '+v.playback_fps+' FPS（显示速率，不是硬件执行速率）\n任务: '+v.instruction;
 clearInterval(timer);timer=setInterval(()=>{if(playing&&frames.length){index=(index+1)%frames.length;el('prediction').src=frames[index];el('frame').textContent=(index+1)+' / '+frames.length}},1000/v.playback_fps);
 }
 if(d.status==='source_unavailable'||d.status==='error'){clearInterval(timer);timer=null;key='';}
 }catch(e){el('status').textContent='连接中断，已停止播放：'+e;clearInterval(timer);timer=null;key='';}
 finally{setTimeout(poll,500)}}poll();
</script></html>'''


def create_app(policy, source_url, gpu_uuid, playback_fps=10., interval=1., fetch_fn=fetch, render_fn=encode_preview):
    state = dict(status='waiting_for_input', error=None, result=None, completed=None)
    async def worker():
        last_key = None
        while True:
            try:
                info = await asyncio.to_thread(fetch_fn,source_url+'/info')
                check_source(info,policy.info,gpu_uuid)
                packet = await asyncio.to_thread(fetch_fn,source_url+'/preview-observation?after='+urllib.parse.quote(last_key or '',safe=''))
                if not packet.get('active'):
                    state.update(status='source_unavailable',error=None)
                elif packet.get('observation') is not None:
                    check_source(dict(packet['resources'],preview_source_enabled=True),policy.info,gpu_uuid)
                    key=packet['key'];last_key=key
                    state.update(status='generating',error=None)
                    result=await asyncio.to_thread(render_fn,policy,packet['observation'])
                    # A disconnected/replaced RTC session must not publish a late result.
                    latest=await asyncio.to_thread(fetch_fn,source_url+'/preview-observation')
                    if not latest.get('active') or latest.get('session')!=packet['session']:
                        state.update(status='source_unavailable',error=None)
                    else:
                        result.update(key=key,session=packet['session'],resources=policy.info,
                            source_age_seconds=packet['source_age_seconds'],playback_fps=playback_fps)
                        state.update(status='ready',error=None,result=result,completed=time.monotonic())
                elif state['result'] is not None:
                    state.update(status='ready',error=None)
            except Exception as e:
                state.update(status='error',error=str(e))
            await asyncio.sleep(interval)  # No action path waits here; no queued video requests.
    @asynccontextmanager
    async def lifespan(app):
        task=asyncio.create_task(worker())
        try:yield
        finally:
            task.cancel()
            await asyncio.gather(task,return_exceptions=True)
    app=FastAPI(lifespan=lifespan)
    @app.get('/',response_class=HTMLResponse)
    async def page():return PAGE
    @app.get('/api/preview')
    async def preview(after: str=''):
        result=state['result']
        return dict(status=state['status'],error=state['error'],
            result_age_seconds=None if state['completed'] is None else time.monotonic()-state['completed'],
            result=result if result and result['key']!=after else None)
    return app


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',required=True)
    p.add_argument('--ckpt-step',type=int,required=True)
    p.add_argument('--cosmos',required=True)
    p.add_argument('--gpu-uuid',required=True,help='Full UUID of a GPU different from RTC')
    p.add_argument('--source-url',default='http://127.0.0.1:8014')
    p.add_argument('--port',type=int,default=8015)
    p.add_argument('--playback-fps',type=float,default=10.,help='Display speed only, not inferred physical time')
    p.add_argument('--interval',type=float,default=1.,help='Idle delay between preview jobs')
    a=p.parse_args(argv)
    source=urllib.parse.urlsplit(a.source_url)
    if source.scheme!='http' or source.hostname!='127.0.0.1' or source.username or source.password or source.path not in ('','/') or source.query or source.fragment:
        p.error('source-url must be a loopback HTTP service')
    if not UUID.fullmatch(a.gpu_uuid):p.error('Use a complete GPU UUID')
    if a.ckpt_step<=0 or not 1<=a.port<=65535:p.error('Invalid checkpoint step/port')
    if not math.isfinite(a.playback_fps) or not 1<=a.playback_fps<=60:p.error('playback-fps must be 1..60')
    if not math.isfinite(a.interval) or a.interval<.1:p.error('interval must be >=0.1 seconds')
    a.source_url=a.source_url.rstrip('/')
    return a


def main():
    a=parse_args()
    info=fetch(a.source_url+'/info')
    check_source(info,None,a.gpu_uuid)
    os.environ['CUDA_VISIBLE_DEVICES']=a.gpu_uuid
    # Set GPU before importing torch/model_adapter. Separate process/model/RNG.
    from model_adapter import Policy
    policy=Policy(a.run_dir,f'steps_{a.ckpt_step}_pytorch_model.pt',a.cosmos)
    check_source(info,policy.info,a.gpu_uuid)
    print('PREVIEW_READY: independent GPU; no motor/action output',flush=True)
    uvicorn.run(create_app(policy,a.source_url,a.gpu_uuid,a.playback_fps,a.interval),host='127.0.0.1',port=a.port)

if __name__=='__main__':main()
