"""Install hash-checked instrumentation, or restore the exact original files.

Does not restart any process. Run while the affected programs are stopped.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--role', choices=['229', '32'], required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--apply', action='store_true', help='Default: only verify and list')
    p.add_argument('--restore', action='store_true')
    a = p.parse_args()
    base = Path(__file__).resolve().parent
    manifest = json.loads((base/'manifest.json').read_text())
    pairs = ([('run_dit4dit_rtc_sonic_dex1.py', 'dit4dit/src/deploy/run_dit4dit_rtc_sonic_dex1.py'),
              ('run_pico_usb_stereo.py', 'gear_sonic/scripts/run_pico_usb_stereo.py')]
             if a.role == '229' else [('serve.py', 'rtc_deploy/serve.py'),
                                     ('model_adapter_32.py', 'rtc_deploy/model_adapter.py')])
    pending = []
    for src, rel in pairs:
        dest = a.root/rel
        expected = manifest[src]['after' if a.restore else 'before']
        desired = manifest[src]['before' if a.restore else 'after']
        if digest(dest) == desired:
            print('Already desired version:', dest)
            continue
        if digest(dest) != expected:
            raise SystemExit('Version differs; no files changed: '+str(dest))
        origin = base/('source_snapshot' if a.restore else 'instrumented')/src
        if digest(origin) != desired:
            raise SystemExit('Bundle checksum failed: '+str(origin))
        pending.append((origin,dest))
    for origin,dest in pending:
        print(('RESTORE' if a.restore else 'INSTALL') if a.apply else 'WOULD CHANGE', dest)
        if a.apply:
            backup = dest.with_suffix(dest.suffix+'.before_latency')
            if not a.restore and not backup.exists():
                shutil.copy2(dest, backup)
            temp = dest.with_suffix(dest.suffix+'.latency_tmp')
            shutil.copy2(origin,temp)
            temp.replace(dest)
    if a.apply and not a.restore:
        # Both -m camera launch and direct deployment scripts can import the SDK.
        sdk_targets = [a.root/'latency_trace.py', a.root/('dit4dit/src/deploy/latency_trace.py'
                                                        if a.role == '229' else 'rtc_deploy/latency_trace.py')]
        for dest in sdk_targets:
            shutil.copy2(base/'latency_trace.py', dest)
    print('No process started or restarted. Add this bundle to PYTHONPATH before launch:', base)


if __name__ == '__main__':
    main()
