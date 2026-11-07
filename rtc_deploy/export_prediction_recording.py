"""Export recorded action chunks and independently generated videos. No robot I/O.

Run against a closed full-diagnostics client recording. Actions are copied from
that recording, never predicted again. Video is separately generated from the
exact recorded observation; video frames are NOT mapped one-to-one to action rows.
"""
import argparse
import base64
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path

IDENTITY = ('checkpoint_sha256', 'config_sha256', 'statistics_sha256')


def read_json(path):
    return json.loads(path.read_text())


def validate_pair(request, chunk, resources):
    import numpy as np
    obs = request['observation']
    for key in ('session', 'request_id'):
        if request.get(key) != chunk.get(key):
            raise ValueError('Request/reply identity mismatch: ' + key)
    for key in ('observation_id', 'observed_at'):
        if obs.get(key) != chunk.get(key):
            raise ValueError('Observation mismatch: ' + key)
    for key in IDENTITY:
        if not resources.get(key) or chunk.get('resources', {}).get(key) != resources[key]:
            raise ValueError('Checkpoint/config/statistics mismatch: ' + key)
    if chunk.get('ok') is not True:
        raise ValueError('Recorded prediction was not successful')
    for key, shape in [('motion_tokens', (50, 64)), ('gripper_actions', (50, 2))]:
        values = np.asarray(chunk[key])
        if values.shape != shape or not np.isfinite(values).all():
            raise ValueError('Invalid recorded ' + key)
    if 'normalized' in chunk:
        values = np.asarray(chunk['normalized'])
        if values.shape != (50, 66) or not np.isfinite(values).all():
            raise ValueError('Invalid normalized actions')
    state = np.asarray(obs['state'])
    if state.shape != (31,) or not np.isfinite(state).all():
        raise ValueError('Invalid observation state')
    return obs


def decode_input(obs):
    from PIL import Image
    if obs.get('image_encoding', 'png') == 'png':
        return Image.open(io.BytesIO(base64.b64decode(obs['image_png'], validate=True))).convert('RGB')
    if obs.get('image_encoding') == 'rgb8_base64' and obs.get('image_shape') == [224,224,3]:
        raw = base64.b64decode(obs['image_rgb'], validate=True)
        if len(raw) == 224*224*3:
            return Image.frombytes('RGB', (224,224), raw)
    raise ValueError('Unsupported input image encoding/shape')


def save_bundle(target, request, chunk, result, events, seed, fps):
    from PIL import Image
    obs = request['observation']
    if (result.get('observation_id'), result.get('observed_at')) != (obs['observation_id'], obs['observed_at']):
        raise ValueError('Rendered video observation mismatch')
    frames = result.get('frames', [])
    if not frames:
        raise ValueError('No generated frames')
    target.mkdir(exist_ok=False)
    (target/'frames').mkdir()
    decode_input(obs).save(target/'input.png')
    images = []
    for i, data in enumerate(frames):
        prefix = 'data:image/jpeg;base64,'
        if not data.startswith(prefix): raise ValueError('Expected generated JPEG frame')
        raw = base64.b64decode(data[len(prefix):], validate=True)
        im = Image.open(io.BytesIO(raw)).convert('RGB');im.load()
        (target/'frames'/f'{i:04d}.jpg').write_bytes(raw)
        images.append(im)
    images[0].save(target/'generated.gif', save_all=True, append_images=images[1:],
                   duration=round(1000/fps), loop=0, optimize=False)
    with (target/'actions.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['action_index']+[f'motion_token_{i}' for i in range(64)]+['left_gripper','right_gripper'])
        for i, (token, grip) in enumerate(zip(chunk['motion_tokens'], chunk['gripper_actions'])):
            writer.writerow([i]+token+grip)
    (target/'prediction.json').write_text(json.dumps(chunk))
    (target/'observation.json').write_text(json.dumps(obs))
    with (target/'delivery_events.jsonl').open('w') as f:
        for event in events: f.write(json.dumps(event)+'\n')
    consumed = sorted({e['index'] for e in events if e.get('kind') in ('delivered','readonly_consumed')})
    metadata = dict(session=chunk['session'], request_id=chunk['request_id'], chunk_id=chunk['chunk_id'],
        observation_id=obs['observation_id'], observed_at=obs['observed_at'], instruction=obs['lang'],
        resources=chunk['resources'], predicted_action_rows=50, consumed_action_indices=consumed,
        motor_execution_confirmed=False, generated_frame_count=len(images), playback_fps=fps,
        playback_is_physical_time=False, video_action_frame_alignment=None,
        video_mode='independent_offline_generation_from_recorded_observation', seed=seed,
        generation_seconds=result.get('generation_seconds'),
        note='Video uses the same input/checkpoint, not the original action forward pass or an executed-action rollout.')
    (target/'metadata.json').write_text(json.dumps(metadata, indent=2))
    return metadata


def export_recording(recording, output, policy, render_fn, fps=10., seed=42, seed_fn=None):
    # Do not use a running recording: graceful client close drains its disk queue.
    session = read_json(recording/'session.json')
    if session.get('diagnostics', 'full') != 'full':
        raise ValueError('Requires --diagnostics full recording; minimal mode omits inputs/actions')
    for key in IDENTITY:
        if session.get('server', {}).get(key) != policy.info.get(key):
            raise ValueError('Recording/model resource mismatch: ' + key)
    requests = {}
    for path in sorted(recording.glob('request_*.json')):
        r = read_json(path); key = (r['session'], r['request_id'])
        if key in requests: raise ValueError('Duplicate request identity')
        requests[key] = r
    chunks = [(p.name, read_json(p)) for p in sorted(recording.glob('chunk_*.json'))]
    warmup = recording/'warmup_discarded.json'
    if warmup.exists(): chunks.append(('warmup_discarded', read_json(warmup)['result']))
    if not chunks: raise ValueError('No recorded action chunks; full diagnostics required')
    events = {}
    event_file = recording/'events.jsonl'
    if event_file.exists():
        with event_file.open() as log:
            for line in log:
                e = json.loads(line)
                if e.get('chunk_id'): events.setdefault(e['chunk_id'], []).append(e)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', bundles=[], failures=[], requests_without_saved_reply=[],
                  video_action_frame_alignment=None, motor_execution_confirmed=False)
    matched = set()
    for index, (name, chunk) in enumerate(chunks):
        try:
            key = (chunk['session'], chunk['request_id'])
            if key in matched: raise ValueError('Duplicate saved reply')
            request = requests[key];matched.add(key)
            obs = validate_pair(request, chunk, policy.info)
            job_seed = (seed + int(hashlib.sha256((key[0]+':'+key[1]).encode()).hexdigest()[:8],16)) % (2**31)
            if seed_fn: seed_fn(job_seed)
            result = render_fn(policy, obs)
            folder = f'{index:06d}'
            metadata = save_bundle(output/folder, request, chunk, result,
                                   events.get(chunk['chunk_id'], []), job_seed, fps)
            report['bundles'].append(dict(directory=folder, source=name, request_id=key[1],
                                          generated_frame_count=metadata['generated_frame_count']))
            print(f'SAVED {folder}: request={key[1]} actions=50 video_frames={metadata["generated_frame_count"]}', flush=True)
        except Exception as exc:
            report['failures'].append(dict(source=name, error=str(exc)))
        (output/'report.json').write_text(json.dumps(report, indent=2))
    report['requests_without_saved_reply'] = [dict(session=k[0],request_id=k[1]) for k in requests if k not in matched]
    report['status'] = 'complete' if not report['failures'] and not report['requests_without_saved_reply'] else 'incomplete'
    (output/'report.json').write_text(json.dumps(report, indent=2))
    return report


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recording', type=Path, required=True, help='Closed client --output directory, diagnostics full')
    p.add_argument('--output', type=Path, required=True, help='New export directory')
    p.add_argument('--run-dir', required=True)
    p.add_argument('--ckpt-step', type=int, required=True)
    p.add_argument('--cosmos', required=True)
    p.add_argument('--gpu', required=True, help='GPU index or full UUID; do not compete with a live robot policy')
    p.add_argument('--playback-fps', type=float, default=10., help='Display rate only; not action/video temporal alignment')
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args(argv)
    if a.ckpt_step <= 0 or not math.isfinite(a.playback_fps) or not 1 <= a.playback_fps <= 60:
        p.error('Invalid checkpoint step or playback FPS')
    if a.output.exists(): p.error('output already exists; choose a new directory')
    return a


def main():
    a = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = a.gpu
    from model_adapter import Policy
    from serve_preview import encode_preview
    import torch
    policy = Policy(a.run_dir, f'steps_{a.ckpt_step}_pytorch_model.pt', a.cosmos, rtc_enabled=False)
    report = export_recording(a.recording, a.output, policy, encode_preview,
                              a.playback_fps, a.seed, torch.manual_seed)
    print('EXPORT_'+report['status'].upper()+': '+str(a.output/'report.json'))
    if report['status'] != 'complete': raise SystemExit(2)

if __name__ == '__main__': main()
