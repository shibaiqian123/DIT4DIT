import unittest
import time
import numpy as np
from fastapi.testclient import TestClient
from serve import create_app

class FakePolicy:
    info={'checkpoint':'fake','rtc_mode':'test_time'}
    def __init__(self):self.prefixes=[]
    def predict(self,obs,previous,d,s):
        self.prefixes.append(previous)
        return dict(normalized=np.ones((50,66)).tolist(),motion_tokens=np.ones((50,64)).tolist(),
                    gripper_actions=np.ones((50,2)).tolist(),inference_seconds=0)

class ServiceTests(unittest.TestCase):
    def request(self,id='r',chunk=None,s=0):
        return dict(op='infer',session='s',request_id=id,chunk_id=chunk,consumed=s,delay=10,
                    observation=dict(observation_id=id,observed_at=1.))
    def test_first_replan_reconnect(self):
        policy=FakePolicy()
        with TestClient(create_app(policy)) as c:
            self.assertFalse(c.get('/health').json()['motor_interfaces'])
            with c.websocket_connect('/ws') as ws:
                ws.send_json(self.request());first=ws.receive_json()
                self.assertTrue(first['ok'])
                ws.send_json(self.request('r2',first['chunk_id'],25));second=ws.receive_json()
                self.assertEqual(second['window_offset'],25)
                self.assertTrue(np.all(policy.prefixes[1][25:]==0))
            with c.websocket_connect('/ws') as ws:
                ws.send_json(self.request());self.assertTrue(ws.receive_json()['ok'])
            self.assertIsNone(policy.prefixes[-1])
    def test_invalid_overlap_fails(self):
        with TestClient(create_app(FakePolicy())) as c,c.websocket_connect('/ws') as ws:
            ws.send_json(self.request());r=ws.receive_json()
            ws.send_json(self.request('r2',r['chunk_id'],49));r=ws.receive_json()
            self.assertFalse(r['ok'])
    def test_slow_call_rejects_concurrent_request(self):
        import threading
        entered=threading.Event();release=threading.Event()
        class Slow(FakePolicy):
            def predict(self,*args):
                entered.set();release.wait(2)
                return super().predict(*args)
        with TestClient(create_app(Slow())) as c,c.websocket_connect('/ws') as ws:
            ws.send_json(self.request())
            self.assertTrue(entered.wait(1))
            ws.send_json(self.request('concurrent'))
            reply=ws.receive_json()
            self.assertFalse(reply['ok'])
            release.set()
            from starlette.websockets import WebSocketDisconnect
            with self.assertRaises(WebSocketDisconnect):
                ws.receive_json()

    def test_unknown_old_chunk(self):
        with TestClient(create_app(FakePolicy())) as c,c.websocket_connect('/ws') as ws:
            ws.send_json(self.request(chunk='stale'));self.assertFalse(ws.receive_json()['ok'])

# Explicit mock endpoint for cross-host transport tests, never a model result.
fixture_app=create_app(FakePolicy())

if __name__=='__main__':unittest.main()
