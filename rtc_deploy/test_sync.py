"""CPU-only sync protocol/service tests; no model, sensors, DDS or motors."""
import unittest
import numpy as np
from fastapi.testclient import TestClient
from serve import create_app, parse_args
from protocol import ConsumptionClock
from test_service import FakePolicy

class PlainPolicy(FakePolicy):
    info = {'rtc_mode': 'disabled'}

class SyncTests(unittest.TestCase):
    def test_launch(self):
        base = ['--run-dir', '/fake', '--ckpt-step', '1', '--cosmos', '/fake']
        a = parse_args(base + ['--sync', '--action_exec_horizon', '16'])
        self.assertFalse(a.rtc_enabled)
        self.assertEqual(a.action_exec_horizon, 50)
        self.assertTrue(parse_args(base).rtc_enabled)
        self.assertFalse(parse_args(base + ['--no-rtc']).rtc_enabled)
        with self.assertRaises(SystemExit): parse_args(base + ['--sync', '--rtc'])

    def test_three_full_blocks_no_skips(self):
        policy = PlainPolicy()
        clock = ConsumptionClock('s', 50, rtc_enabled=False, synchronous=True)
        with TestClient(create_app(policy, 50, synchronous=True)) as c, c.websocket_connect('/ws') as ws:
            self.assertEqual(c.get('/info').json()['execution_mode'], 'sync')
            for b in range(3):
                self.assertTrue(clock.ready_to_request())
                r = clock.request(str(b))
                self.assertEqual(r['consumed'], 0 if b == 0 else 50)
                self.assertEqual(r['delay'], 0)
                self.assertFalse(clock.ready_to_request())
                ws.send_json(dict(r, op='infer', observation=dict(observation_id=str(b), observed_at=1.)))
                self.assertTrue(clock.accept(ws.receive_json()))
                self.assertEqual(clock.position, 0)
                for i in range(50):
                    self.assertFalse(clock.ready_to_request())
                    self.assertEqual(clock.consume(), i)
            self.assertEqual(clock.total, 150)
            self.assertEqual(policy.prefixes, [None, None, None])

    def test_sync_rejects_early_and_wrong_mode(self):
        for consumed, mode in [(49, 'sync'), (50, 'async')]:
            with self.subTest(consumed=consumed, mode=mode):
                with TestClient(create_app(PlainPolicy(), 50, synchronous=True)) as c, c.websocket_connect('/ws') as ws:
                    r = dict(op='infer', session='s', request_id='a', chunk_id=None, consumed=0,
                             delay=0, execution_mode='sync', observation=dict(observation_id='o', observed_at=1.))
                    ws.send_json(r); first = ws.receive_json()
                    ws.send_json(dict(r, request_id='b', chunk_id=first['chunk_id'], consumed=consumed, execution_mode=mode))
                    self.assertFalse(ws.receive_json()['ok'])

    def test_async_ordinary_still_replans_and_skips(self):
        clock = ConsumptionClock('s', 16, rtc_enabled=False)
        with TestClient(create_app(PlainPolicy(), 16)) as c, c.websocket_connect('/ws') as ws:
            for b in range(2):
                r = clock.request(str(b))
                ws.send_json(dict(r, op='infer', observation=dict(observation_id=str(b), observed_at=1.)))
                if b:
                    for _ in range(3): clock.consume()
                self.assertTrue(clock.accept(ws.receive_json()))
                self.assertEqual(clock.position, 0 if b == 0 else 3)
                if not b:
                    for _ in range(16): clock.consume()
                    self.assertTrue(clock.ready_to_request())

    def test_selected_steps_contract(self):
        base = ['--run-dir', '/fake', '--ckpt-step', '1', '--cosmos', '/fake']
        for n in (1, 25, 30, 50):
            with self.subTest(n=n):
                self.assertEqual(parse_args(base + ['--sync', '--sync-steps', str(n)]).action_exec_horizon, n)
                clock = ConsumptionClock('s', n, rtc_enabled=False, synchronous=True)
                with TestClient(create_app(PlainPolicy(), n, synchronous=True)) as c, c.websocket_connect('/ws') as ws:
                    self.assertEqual(c.get('/info').json()['action_exec_horizon_default'], n)
                    for b in range(2):
                        r = clock.request(str(b))
                        self.assertEqual(r['consumed'], 0 if b == 0 else n)
                        ws.send_json(dict(r, op='infer', observation=dict(observation_id=str(b), observed_at=1.)))
                        self.assertTrue(clock.accept(ws.receive_json()))
                        self.assertEqual(clock.position, 0)
                        for i in range(n):
                            self.assertFalse(clock.ready_to_request())
                            self.assertEqual(clock.consume(), i)
                        self.assertTrue(clock.ready_to_request())
                    with self.assertRaises(ValueError): clock.consume()

    def test_invalid_selected_steps(self):
        import contextlib
        import io
        base = ['--run-dir', '/fake', '--ckpt-step', '1', '--cosmos', '/fake']
        for args in (['--sync', '--sync-steps', '0'], ['--sync', '--sync-steps', '51'],
                     ['--sync', '--sync-steps', '1.5'], ['--sync-steps', '25']):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(base + args)
        for n in (0, 51, True):
            with self.subTest(n=n), self.assertRaises(ValueError):
                ConsumptionClock('s', n, rtc_enabled=False, synchronous=True)

    def test_selected_steps_server_rejects_wrong_boundary(self):
        for n in (25, 30):
            for consumed in (n-1, n+1, 50):
                with self.subTest(n=n, consumed=consumed):
                    with TestClient(create_app(PlainPolicy(), n, synchronous=True)) as c, c.websocket_connect('/ws') as ws:
                        r = dict(op='infer', session='s', request_id='a', chunk_id=None, consumed=0,
                                 delay=0, execution_mode='sync', observation=dict(observation_id='o', observed_at=1.))
                        ws.send_json(r); first = ws.receive_json()
                        ws.send_json(dict(r, request_id='b', chunk_id=first['chunk_id'], consumed=consumed))
                        self.assertFalse(ws.receive_json()['ok'])

    def test_mode_and_exhaustion_checks(self):
        with self.assertRaises(ValueError): ConsumptionClock('s', 50, synchronous=True)
        with self.assertRaises(ValueError): create_app(FakePolicy(), 50, synchronous=True)
        clock = ConsumptionClock('s', 50, rtc_enabled=False, synchronous=True)
        with self.assertRaises(ValueError): clock.consume()

if __name__ == '__main__': unittest.main()
