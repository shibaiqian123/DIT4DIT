import importlib
import json
import os
from pathlib import Path
import socket
import unittest
import latency_trace as t


def event(stage, mono, host='229', wall=None, tid='request1'):
    return dict(event_id=stage+host, trace_id=tid, stage=stage,
                clock=dict(host=host, boot='boot'+host, mono_ns=mono*1000000,
                           wall_ns=(mono if wall is None else wall)*1000000))


class TimingTest(unittest.TestCase):
    def test_monotonic_ignores_wall_jump(self):
        self.assertEqual(t.duration(event('a',10,wall=9000),event('b',25,wall=0),{})[0],15)

    def test_unsynchronized_never_fabricates_one_way(self):
        self.assertIsNone(t.duration(event('a',0),event('b',100,'32'),{})[0])

    def test_bound_and_expiry(self):
        sync={h:dict(offset_ms=o,uncertainty_ms=.5,valid_from_ns=0,valid_until_ns=1000000000)
              for h,o in [('229',0),('32',20)]}
        self.assertEqual(t.duration(event('a',0),event('b',50,'32'),sync),(30,1,'synced_epoch'))
        self.assertEqual(t.duration(event('a',0),event('b',2000,'32'),sync)[2],'sync_evidence_expired')

    def test_roundtrip_residual(self):
        es=[event('request_send_start',0),event('reply_received',400),
            event('server_received',900,'32'),event('reply_send_start',1200,'32')]
        rows=t.analyze(es)
        self.assertEqual(next(r for r in rows if r['segment']=='transport_and_serialization_roundtrip')['ms'],100)

    def test_missing_and_duplicates(self):
        a=event('request_send_start',0)
        b=event('reply_received',10)
        c=dict(b,event_id='other')
        rows=t.analyze([a,b,c])
        self.assertEqual(next(r for r in rows if r['segment']=='request_roundtrip_229')['status'],'ambiguous_duplicate')
        rows=t.analyze([a,b,b])
        self.assertEqual(next(r for r in rows if r['segment']=='request_roundtrip_229')['ms'],10)
        self.assertTrue(any(r['status']=='missing_event' for r in rows))

    def test_no_cross_boot_monotonic(self):
        a,b=event('a',0),event('b',100)
        b['clock']['boot']='reboot'
        self.assertIsNone(t.duration(a,b,{})[0])

    def test_udp_sdk(self):
        s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        s.bind(('127.0.0.1',0)); s.settimeout(1)
        os.environ['LATENCY_TRACE_UDP']='127.0.0.1:'+str(s.getsockname()[1])
        importlib.reload(t)
        t.mark('unique','probe')
        e=json.loads(s.recv(8192))
        self.assertEqual(e['trace_id'],'unique')
        s.close(); t._sock.close()
        os.environ.pop('LATENCY_TRACE_UDP')
        importlib.reload(t)

    def test_instrumented_files_compile(self):
        for p in (Path(__file__).parent/'instrumented').glob('*.py'):
            compile(p.read_text(),str(p),'exec')


if __name__=='__main__':
    unittest.main()
