"""Read-only chrony evidence; does not set time or change chrony configuration."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
import uuid
from latency_trace import stamp, mark

def parse_tracking(text):
    fields = {}
    for line in text.splitlines():
        if ':' in line:
            key, value = line.split(':', 1)
            fields[key.strip()] = value.strip()
    if fields.get('Leap status') != 'Normal':
        raise ValueError('chrony is not normally synchronized')
    ref = fields.get('Reference ID', '').split()[0]
    if ref in ('00000000', '7F7F0101', '127.127.1.1'):
        raise ValueError('No external reference')
    stratum = int(fields['Stratum'])
    if not 1 <= stratum <= 15:
        raise ValueError('Invalid stratum')
    match = re.fullmatch(r'([\d.eE+-]+) seconds (fast|slow) of NTP time', fields['System time'])
    if not match:
        raise ValueError('Unknown System time format')
    offset = float(match[1]) * (1 if match[2] == 'fast' else -1)
    delay = float(fields['Root delay'].split()[0])
    dispersion = float(fields['Root dispersion'].split()[0])
    skew = float(fields['Skew'].split()[0])
    if not all(math.isfinite(x) for x in (offset, delay, dispersion, skew)) or min(delay,dispersion,skew)<0:
        raise ValueError('Invalid chrony accuracy fields')
    # chronyc documented bound, conditional on a correct stratum-1 reference.
    bound_ms = 1000 * (abs(offset) + dispersion + delay/2)
    return dict(reference=ref, stratum=stratum, measured_offset_ms=offset*1000,
                uncertainty_ms=bound_ms, skew_ppm=skew, reference_scale='UTC')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--interval', type=float, default=2)
    p.add_argument('--once', action='store_true')
    a=p.parse_args()
    if not 0.1 <= a.interval <= 60:
        p.error('interval must be 0.1..60 seconds')
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    with open(a.output,'a',buffering=1) as f:
        try:
            while True:
                start=stamp()
                raw=''
                try:
                    result=subprocess.run(['chronyc','-n','tracking'],text=True,capture_output=True,
                                          timeout=2,env=dict(os.environ,LC_ALL='C'))
                    raw=result.stdout
                    if result.returncode:
                        raise ValueError(result.stderr.strip() or 'chronyc failed')
                    estimate=parse_tracking(raw)
                    status='ok'
                except (OSError,ValueError,KeyError,IndexError,subprocess.TimeoutExpired) as e:
                    estimate={}
                    status=str(e)
                end=stamp()
                event=dict(v=1,event_id=uuid.uuid4().hex,trace_id='clock:'+uuid.uuid4().hex,
                           stage='clock_sync',clock=end,meta=dict(status=status,estimate=estimate,
                           query_start=start,raw_tracking=raw),emitter_drops=0)
                f.write(json.dumps(event)+'\n')
                print(status, estimate, flush=True)
                # Emit the same evidence; file is authoritative if UDP is dropped.
                mark(event['trace_id'], 'clock_sync', at=end, **event['meta'])
                if a.once:break
                time.sleep(a.interval)
        except KeyboardInterrupt:
            pass
if __name__=='__main__':main()
