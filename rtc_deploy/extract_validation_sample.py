"""Extract dataset episode0/frame0 as a traceable offline fixture; not a live capture."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import cv2
p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
parquet=a.dataset/'data/chunk-000/episode_000000.parquet'
video=a.dataset/'videos/chunk-000/observation.images.ego_view/episode_000000.mp4'
row=pq.read_table(parquet).slice(0,1).to_pylist()[0]
state=np.asarray(row['observation.sonic_state'])
if state.shape!=(31,) or not np.isfinite(state).all():raise ValueError('Invalid recorded state')
cap=cv2.VideoCapture(str(video));ok,img=cap.read();cap.release()
if not ok:raise ValueError('Video first frame missing')
image=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
task={r['task_index']:r['task'] for r in map(json.loads,(a.dataset/'meta/tasks.jsonl').read_text().splitlines())}[row['task_index']]
a.output.mkdir(parents=True,exist_ok=False)
np.savez(a.output/'observation.npz',state=state,image=image,observed_at=float(row['timestamp']))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
(a.output/'manifest.json').write_text(json.dumps(dict(task_prompt=task,files={'observation.npz':sha(a.output/'observation.npz')},
 source=dict(dataset=str(a.dataset),parquet=str(parquet),parquet_sha256=sha(parquet),video=str(video),video_sha256=sha(video),
 episode=0,frame=0,state_key='observation.sonic_state',timestamp_semantics='dataset_relative',live_capture=False)),indent=2))
print('DATASET_FIXTURE_SAVED',a.output)
