import unittest
import numpy as np
from fastapi.testclient import TestClient
from serve import parse_args,create_app
from test_service import FakePolicy
import test_service
class LaunchTests(unittest.TestCase):
    def test_aliases(self):
        a=parse_args(['--host','0.0.0.0','--port','8014','--action_exec_horizon','30',
                     '--policy','dit4dit','--rtc','--run-dir','/run','--ckpt-step','19000','--cosmos','/cosmos'])
        self.assertEqual(a.checkpoint,'steps_19000_pytorch_model.pt')
        self.assertEqual(a.action_exec_horizon,30)
    def test_window_enforced_and_advertised(self):
        req=test_service.ServiceTests().request
        with TestClient(create_app(FakePolicy(),30)) as c,c.websocket_connect('/ws') as ws:
            self.assertEqual(c.get('/info').json()['action_exec_horizon_default'],30)
            ws.send_json(req());r=ws.receive_json()
            ws.send_json(req('next',r['chunk_id'],25));self.assertFalse(ws.receive_json()['ok'])
    def test_window30_accepted(self):
        req=test_service.ServiceTests().request
        with TestClient(create_app(FakePolicy(),30)) as c,c.websocket_connect('/ws') as ws:
            ws.send_json(req());r=ws.receive_json()
            ws.send_json(req('next',r['chunk_id'],30));self.assertTrue(ws.receive_json()['ok'])
if __name__=='__main__':unittest.main()
