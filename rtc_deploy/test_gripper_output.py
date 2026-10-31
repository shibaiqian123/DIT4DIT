"""Offline adapter tests: no checkpoint, CUDA execution or robot connection."""
from types import SimpleNamespace
import unittest
from contextlib import nullcontext
from unittest.mock import patch

import numpy as np
import torch

from model_adapter import Policy, binary_gripper_mode, gripper_motor_targets, DEX1_OPEN_TARGETS


def statistics(binary=True):
    result = dict(min=[-2.] * 64 + [0., 0.], max=[2.] * 64 + [1., 1.])
    if binary:
        result['dex1_label_encoding'] = dict(source='teleop.dex1_trigger', open=0, closed=1)
    return result


class GripperOutputTests(unittest.TestCase):
    def test_threshold_and_side_order(self):
        actions = np.zeros((50, 66))
        actions[:4, -2:] = [[0., 1.], [.4999, .5], [.5, .4999], [1., 0.]]
        before = actions.copy()
        targets = gripper_motor_targets(actions, statistics())
        left, right = DEX1_OPEN_TARGETS
        np.testing.assert_array_equal(targets[:4], [[left, 0.], [left, 0.], [0., right], [0., right]])
        np.testing.assert_array_equal(actions, before)

    def test_legacy_continuous_is_unchanged(self):
        actions = np.linspace(-2, 5.3, 50 * 66).reshape(50, 66)
        np.testing.assert_array_equal(gripper_motor_targets(actions, statistics(False)), actions[:, -2:])
        self.assertFalse(binary_gripper_mode(statistics(False), False))

    def test_bad_contracts_fail(self):
        for stats, configured in [(statistics(), False), (statistics(False), True)]:
            with self.assertRaises(ValueError):
                binary_gripper_mode(stats, configured)
        for field, value in [('open', 1), ('closed', 0), ('source', 'observation.dex1_position')]:
            stats = statistics()
            stats['dex1_label_encoding'][field] = value
            with self.assertRaises(ValueError):
                binary_gripper_mode(stats)
        for key, value in [('min', -1.), ('max', 5.3)]:
            stats = statistics()
            stats[key][-1] = value
            with self.assertRaises(ValueError):
                binary_gripper_mode(stats)

    def test_nonfinite_and_out_of_range_fail(self):
        for value in [np.nan, np.inf, -.01, 1.01]:
            actions = np.zeros((50, 66))
            actions[0, -1] = value
            with self.assertRaises(ValueError):
                gripper_motor_targets(actions, statistics())

    def test_predict_preserves_normalized_rtc_and_tokens(self):
        # Run the actual adapter predict method for both initial and RTC paths.
        for binary in (False, True):
            norm = np.linspace(-1, 1, 50 * 66, dtype=np.float32).reshape(50, 66)
            policy = Policy.__new__(Policy)
            policy.stats = {'action': statistics(binary)}
            policy.example = lambda obs: dict(image=[], lang='test', state=np.zeros((1,31), dtype=np.float32))
            def unnormalize(values, stats):
                lo, hi = np.array(stats['min']), np.array(stats['max'])
                return (values + 1) * .5 * (hi - lo) + lo
            policy.unnormalize = unnormalize
            class Backbone:
                def build_cosmos_inputs(self, **kw):
                    return {}
                def __call__(self, **kw):
                    return SimpleNamespace(hidden_states=[torch.zeros((1,1,1))])
            policy.model = SimpleNamespace(
                predict_action=lambda examples: {'normalized_actions': norm[None].copy()},
                backbone_interface=Backbone())
            saved = []
            class RTC:
                def predict_action_rtc(self, features, state, previous, *args):
                    saved.append(previous.copy())
                    return torch.from_numpy(norm[None].copy())
            policy.rtc = RTC()
            previous = norm.copy()
            for prior in (None, previous):
                with patch('torch.cuda.synchronize'), patch('torch.autocast', return_value=nullcontext()):
                    result = policy.predict({}, previous=prior)
                np.testing.assert_array_equal(result['normalized'], norm)
                actions = unnormalize(norm, policy.stats['action'])
                np.testing.assert_array_equal(result['motion_tokens'], actions[:, :64])
                np.testing.assert_array_equal(result['gripper_actions'], gripper_motor_targets(actions, policy.stats['action']))
            np.testing.assert_array_equal(previous, norm)
            np.testing.assert_array_equal(saved[0], norm[None])


if __name__ == '__main__':
    unittest.main()
