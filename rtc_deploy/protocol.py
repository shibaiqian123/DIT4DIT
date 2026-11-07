"""Pure client-consumption clock. No DDS, ZMQ command or motor interfaces."""
from collections import deque
import numpy as np

H, D, HZ = 50, 66, 50


def block(value):
    a = np.asarray(value, dtype=np.float32)
    if a.shape != (H, D) or not np.isfinite(a).all():
        raise ValueError('Expected finite normalized [50,66] block')
    return a


def shifted(previous, consumed, delay):
    if type(consumed) is not int or type(delay) is not int or not (0 <= delay <= consumed <= H-delay) or consumed >= H:
        raise ValueError('Insufficient history / invalid RTC overlap')
    a = block(previous)
    return np.concatenate((a[consumed:], np.zeros((consumed, D), dtype=a.dtype)))


def rtc_enabled_from_info(info):
    mode = info.get('rtc_mode')
    if mode not in ('test_time', 'disabled'):
        raise ValueError('Unknown server RTC mode')
    enabled = mode == 'test_time'
    if 'rtc_enabled' in info and info['rtc_enabled'] is not enabled:
        raise ValueError('Inconsistent server RTC mode')
    return enabled


class ConsumptionClock:
    def __init__(self, session, horizon=25, *, allow_compact=False, rtc_enabled=True, synchronous=False):
        if synchronous and rtc_enabled:
            raise ValueError("Synchronous execution requires ordinary sampling")
        if type(horizon) is not int or not (1 <= horizon <= H if synchronous else 1 <= horizon < H):
            raise ValueError('Invalid action_exec_horizon')
        self.synchronous = synchronous
        self.rtc_enabled = rtc_enabled
        self.allow_compact = allow_compact
        self.session, self.horizon = session, horizon
        self.chunk_id = None
        self.position = 0
        self.total = 0
        self.pending = None
        self.failed = False
        self.delays = deque([10], maxlen=6)

    def ready_to_request(self):
        """Wait for enough consumed history; never relax the overlap bounds."""
        if self.failed:
            raise ValueError('Failed session')
        if self.pending is not None:
            return False
        if self.chunk_id is None:
            return True
        if self.synchronous:
            return self.position == self.horizon
        if not self.rtc_enabled:
            if self.position >= H:
                raise ValueError('No valid action remains')
            return self.position >= self.horizon
        delay = max(self.delays)
        earliest = max(self.horizon, delay)
        latest = min(H-delay, H-1)
        if earliest > latest or self.position > latest:
            raise ValueError(f'No feasible RTC overlap: position={self.position}, '
                             f'delay={delay}, horizon={self.horizon}, latest={latest}')
        return self.position >= earliest

    def request(self, request_id):
        if self.failed or self.pending is not None:
            raise ValueError('Failed session or request already in flight')
        if self.chunk_id is not None and self.position < self.horizon:
            raise ValueError('Replan before execution horizon')
        if self.synchronous and self.chunk_id is not None and self.position != self.horizon:
            raise ValueError('Synchronous request must follow the configured number of rows')
        if self.chunk_id is not None and self.position >= H and not (self.synchronous and self.position == self.horizon):
            raise ValueError('No valid action remains')
        d = 0 if self.synchronous else max(self.delays)
        if self.rtc_enabled and self.chunk_id is not None:
            shifted(np.zeros((H,D)), self.position, d)
        req = dict(session=self.session, request_id=request_id, chunk_id=self.chunk_id,
                   consumed=self.position, delay=d)
        if self.synchronous:
            req['execution_mode'] = 'sync'
        self.pending = (request_id, self.total, self.chunk_id, self.position)
        return req

    def accept(self, result):
        if result.get('session') != self.session:
            return False  # old connection result cannot poison a new session
        if self.failed:
            raise ValueError('Session already failed')
        if self.pending is None or result.get('request_id') != self.pending[0]:
            raise ValueError('Unexpected reply identity')
        if not result.get('ok'):
            self.failed = True
            raise ValueError(result.get('error', 'Inference failed'))
        mode = result.get('resources', {}).get('rtc_mode')
        expected = 'test_time' if self.rtc_enabled else 'disabled'
        if (mode is not None and mode != expected) or (not self.rtc_enabled and mode is None):
            self.failed = True
            raise ValueError('Reply RTC mode changed or missing')
        if result.get('window_offset') != self.pending[3]:
            self.failed = True
            raise ValueError('Reply window offset mismatch')
        if not isinstance(result.get('chunk_id'), str) or not result['chunk_id'] or result['chunk_id'] == self.pending[2]:
            self.failed = True
            raise ValueError('Invalid new chunk identity')
        lag = self.total-self.pending[1]
        if self.synchronous and (lag != 0 or result.get('resources', {}).get('execution_mode') != 'sync'):
            raise ValueError('Synchronous reply mode/progress mismatch')
        if lag >= H:
            self.failed = True
            raise ValueError('Returned chunk already expired')
        if result.get('reply_format') == 'actions_only_v1':
            if not self.allow_compact or result.get('normalized_validated') is not True:
                raise ValueError('Unnegotiated compact reply or missing server validation')
            token = np.asarray(result.get('motion_tokens'), dtype=np.float32)
            grip = np.asarray(result.get('gripper_actions'), dtype=np.float64)
            if (token.shape != (H,64) or grip.shape != (H,2)
                    or not np.isfinite(token).all() or not np.isfinite(grip).all()):
                raise ValueError('Invalid compact action block')
        else:
            block(result['normalized'])
        if self.chunk_id is not None:
            self.delays.append(lag)
        self.chunk_id = result['chunk_id']
        self.position = lag
        self.pending = None
        return True

    def consume(self):
        limit = self.horizon if self.synchronous else H
        if self.failed or self.chunk_id is None or self.position >= limit:
            self.failed = True
            raise ValueError('No valid action remains; no last-token repetition')
        index = self.position
        self.position += 1
        self.total += 1
        return index
