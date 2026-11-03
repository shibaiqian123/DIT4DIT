import unittest
from clock_watch import parse_tracking
from latency_trace import clock_intervals, duration, analyze
from test_latency_trace import event

TRACK="""Reference ID : C0A80101 (192.168.1.1)
Stratum : 3
System time : 0.0001 seconds slow of NTP time
Root delay : 0.002 seconds
Root dispersion : 0.0002 seconds
Skew : 0.1 ppm
Leap status : Normal
"""

def clock_sample(host, mono, wall, status='ok'):
    e=event('clock_sync',mono,host,wall,tid=host+str(mono))
    e['event_id']=host+str(mono)
    e['meta']=dict(status=status,estimate=parse_tracking(TRACK),query_start=dict(e['clock']))
    return e

class ClockTests(unittest.TestCase):
    def test_parse_bound_and_sign(self):
        r=parse_tracking(TRACK)
        self.assertAlmostEqual(r['measured_offset_ms'],-.1)
        self.assertAlmostEqual(r['uncertainty_ms'],1.3)

    def test_unsync_rejected(self):
        with self.assertRaises(ValueError):parse_tracking(TRACK.replace('Normal','Not synchronised'))
        with self.assertRaises(ValueError):parse_tracking(TRACK.replace('C0A80101','7F7F0101'))

    def test_brackets_required_and_steps_rejected(self):
        self.assertFalse(clock_intervals([clock_sample('229',0,0)]))
        self.assertFalse(clock_intervals([clock_sample('229',0,0),clock_sample('229',2000,2200)]))
        self.assertFalse(clock_intervals([clock_sample('229',0,0),clock_sample('229',20000,20000)]))

    def test_auto_sync_and_expiry(self):
        es=[clock_sample('229',0,0),clock_sample('229',2000,2000),
            clock_sample('32',0,0),clock_sample('32',2000,2000)]
        es += [event('request_send_start',500,'229'),event('server_received',520,'32')]
        rows=analyze(es)
        r=next(r for r in rows if r['segment']=='uplink_229_to_32')
        self.assertEqual(r['ms'],20)
        self.assertGreater(r['uncertainty_ms'],2)
        self.assertIsNone(duration(event('a',3000,'229'),event('b',3020,'32'),clock_intervals(es))[0])

if __name__=='__main__':unittest.main()
