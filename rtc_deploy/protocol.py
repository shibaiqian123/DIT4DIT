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


class ConsumptionClock:
    def __init__(self, session, horizon=25):
        if type(horizon) is not int or not 1 <= horizon < H:
            raise ValueError('Invalid action_exec_horizon')
        self.session, self.horizon = session, horizon
        self.chunk_id = None
        self.position = 0
        self.total = 0
        self.pending = None
        self.failed = False
        self.delays = deque([10], maxlen=6)

    def request(self, request_id):
        if self.failed or self.pending is not None:
            raise ValueError('Failed session or request already in flight')
        if self.chunk_id is not None and self.position < self.horizon:
            raise ValueError('Replan before execution horizon')
        d = max(self.delays)
        if self.chunk_id is not None:
            shifted(np.zeros((H,D)), self.position, d)
        req = dict(session=self.session, request_id=request_id, chunk_id=self.chunk_id,
                   consumed=self.position, delay=d)
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
        if result.get('window_offset') != self.pending[3]:
            self.failed = True
            raise ValueError('Reply window offset mismatch')
        if not isinstance(result.get('chunk_id'), str) or not result['chunk_id'] or result['chunk_id'] == self.pending[2]:
            self.failed = True
            raise ValueError('Invalid new chunk identity')
        lag = self.total-self.pending[1]
        if lag >= H:
            self.failed = True
            raise ValueError('Returned chunk already expired')
        block(result['normalized'])
        if self.chunk_id is not None:
            self.delays.append(lag)
        self.chunk_id = result['chunk_id']
        self.position = lag
        self.pending = None
        return True

    def consume(self):
        if self.failed or self.chunk_id is None or self.position >= H:
            self.failed = True
            raise ValueError('No valid action remains; no last-token repetition')
        index = self.position
        self.position += 1
        self.total += 1
        return index
