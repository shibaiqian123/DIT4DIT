import base64
import csv
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from PIL import Image
import numpy as np
from export_prediction_recording import export_recording, validate_pair, decode_input

class RecordingTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.recording=self.root/'recording';self.recording.mkdir()
        self.resources={k:'a'*64 for k in ('checkpoint_sha256','config_sha256','statistics_sha256')}
        self.policy=SimpleNamespace(info=self.resources)
        b=io.BytesIO();Image.new('RGB',(224,224),(10,20,30)).save(b,format='PNG')
        self.obs=dict(image_png=base64.b64encode(b.getvalue()).decode(),state=[0.]*31,lang='test',
                      observation_id='obs',observed_at=10.)
        self.request=dict(session='session',request_id='request',observation=self.obs)
        self.chunk=dict(ok=True,session='session',request_id='request',chunk_id='chunk',
                        observation_id='obs',observed_at=10.,resources=self.resources,
                        motion_tokens=np.zeros((50,64)).tolist(),gripper_actions=np.zeros((50,2)).tolist())
        self.write('session.json',dict(server=self.resources,diagnostics='full'))
        self.write('request_request.json',self.request);self.write('chunk_0001.json',self.chunk)
        (self.recording/'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in [
            dict(kind='delivered',chunk_id='chunk',index=0),dict(kind='delivered',chunk_id='chunk',index=1),
            dict(kind='sync_wait_target',chunk_id='chunk',index=1)]))
    def write(self,name,data): (self.recording/name).write_text(json.dumps(data))
    def render(self,policy,obs):
        frames=[]
        for color in ('red','blue'):
            b=io.BytesIO();Image.new('RGB',(224,224),color).save(b,format='JPEG')
            frames.append('data:image/jpeg;base64,'+base64.b64encode(b.getvalue()).decode())
        return dict(frames=frames,observation_id=obs['observation_id'],observed_at=obs['observed_at'])
    def run_export(self,**kwargs):
        return export_recording(self.recording,self.root/'out',self.policy,self.render,**kwargs)
    def test_complete_bundle_and_consumption(self):
        seeds=[];r=self.run_export(seed_fn=seeds.append)
        self.assertEqual(r['status'],'complete');self.assertEqual(len(seeds),1)
        bundle=self.root/'out'/'000000'
        with (bundle/'actions.csv').open() as f: rows=list(csv.reader(f))
        self.assertEqual(len(rows),51);self.assertEqual(len(rows[0]),67)
        self.assertEqual(len(list((bundle/'frames').glob('*.jpg'))),2)
        with Image.open(bundle/'generated.gif') as im:self.assertEqual(im.n_frames,2)
        m=json.loads((bundle/'metadata.json').read_text())
        self.assertEqual(m['consumed_action_indices'],[0,1])
        self.assertIsNone(m['video_action_frame_alignment'])
        self.assertFalse(m['motor_execution_confirmed'])
        self.assertEqual(json.loads((bundle/'prediction.json').read_text()),self.chunk)
    def test_wrong_observation_is_reported(self):
        self.chunk['observation_id']='wrong';self.write('chunk_0001.json',self.chunk)
        r=self.run_export();self.assertEqual(r['status'],'incomplete');self.assertEqual(len(r['failures']),1)
        self.assertEqual(r['bundles'],[])
    def test_missing_reply_is_reported(self):
        self.write('request_other.json',dict(self.request,request_id='other'))
        r=self.run_export();self.assertEqual(r['status'],'incomplete')
        self.assertEqual(r['requests_without_saved_reply'],[dict(session='session',request_id='other')])
    def test_minimal_and_wrong_model_rejected(self):
        self.write('session.json',dict(server=self.resources,diagnostics='minimal'))
        with self.assertRaises(ValueError):self.run_export()
        self.write('session.json',dict(server={},diagnostics='full'))
        with self.assertRaises(ValueError):self.run_export()
    def test_warmup_not_silently_lost(self):
        r=dict(self.request,request_id='warmup');c=dict(self.chunk,request_id='warmup',chunk_id='warmup')
        self.write('request_warmup.json',r);self.write('warmup_discarded.json',dict(result=c))
        result=self.run_export();self.assertEqual(result['status'],'complete');self.assertEqual(len(result['bundles']),2)
    def test_rgb_input_and_invalid_shape(self):
        obs=dict(self.obs,image_encoding='rgb8_base64',image_shape=[224,224,3],image_rgb=base64.b64encode(bytes(224*224*3)).decode())
        self.assertEqual(decode_input(obs).size,(224,224))
        chunk=dict(self.chunk,motion_tokens=[[0.]*64])
        with self.assertRaises(ValueError):validate_pair(self.request,chunk,self.resources)
    def test_render_error_retains_incomplete_report(self):
        def fail(*args):raise RuntimeError('GPU test failure')
        r=export_recording(self.recording,self.root/'out',self.policy,fail)
        self.assertEqual(r['status'],'incomplete')
        self.assertIn('GPU test failure',r['failures'][0]['error'])

if __name__=='__main__':unittest.main()
