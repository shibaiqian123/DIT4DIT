"""CPU checks; optionally set DEX1_TEST_DATASET for real-episode raw-loader checks."""
import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from DiT4DiT.dataloader.gr00t_lerobot.dex1_binary import trigger_to_binary, episode_labels, label_statistics


class BinaryLabelsTest(unittest.TestCase):
    def test_hysteresis_sides_and_reset(self):
        trigger = np.array([[0., 1.], [.8, .2], [.5, .5], [.2, .8], [.5, .5]])
        np.testing.assert_array_equal(trigger_to_binary(trigger, np.ones_like(trigger)),
                                      [[0, 1], [1, 0], [1, 0], [0, 1], [0, 1]])
        np.testing.assert_array_equal(trigger_to_binary([[.4, .6]], [[1, 1]]), [[0, 1]])
        np.testing.assert_array_equal(trigger_to_binary([[.5, .5]], [[1, 1]]), [[1, 1]])

    def test_invalid_inputs_fail(self):
        for trigger, valid in [([[np.nan, 0]], [[1, 1]]), ([[1.1, 0]], [[1, 1]]),
                               ([[0, 0]], [[0, 1]]), ([], []), ([[0]], [[1]])]:
            with self.assertRaises(ValueError):
                trigger_to_binary(trigger, valid)
        with self.assertRaises(ValueError):
            trigger_to_binary([[0, 1]], [[1, 1]], .8, .2)


def raw_loader_harness(root):
    # Execute the actual raw-read and padding methods without importing VLM/video dependencies.
    source = Path(__file__).resolve().parents[1] / 'DiT4DiT/dataloader/gr00t_lerobot/datasets.py'
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'LeRobotSingleDataset')
    names = {'get_state_or_action', 'retrieve_data_and_pad'}
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = {'np': np, '__package__': 'DiT4DiT.dataloader.gr00t_lerobot'}
    exec(compile(ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[])), str(source), 'exec'), namespace)
    harness = type('RawLoaderHarness', (), {name: namespace[name] for name in names})()
    modality = json.loads((root / 'meta/modality.json').read_text())
    harness.lerobot_modality_meta = SimpleNamespace(**{
        kind: {key: SimpleNamespace(**value) for key, value in modality[kind].items()}
        for kind in ('action', 'state')})
    harness.metadata = SimpleNamespace(modalities=harness.lerobot_modality_meta)
    harness.delta_indices = {kind + '.' + key: np.arange(50) if kind == 'action' else np.array([0])
                             for kind in ('action', 'state') for key in modality[kind]}
    harness.get_trajectory_index = lambda tid: 0
    harness.dex1_open_threshold = .2
    harness.dex1_close_threshold = .8
    harness._dex1_binary_episode_id = None
    return harness


@unittest.skipUnless(os.environ.get('DEX1_TEST_DATASET'), 'Set DEX1_TEST_DATASET to check real episodes')
class RealEpisodeTest(unittest.TestCase):
    def test_raw_windows_stats_and_source_unchanged(self):
        import pandas as pd
        root = Path(os.environ['DEX1_TEST_DATASET'])
        harness = raw_loader_harness(root)
        paths = sorted(root.glob('data/*/*.parquet'))
        tracked = paths + list((root / 'meta').glob('*'))
        before = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in tracked}
        all_labels = []
        windows = 0
        for episode, path in enumerate(paths):
            frame = pd.read_parquet(path)
            labels = episode_labels(frame)
            all_labels.append(labels)
            body = np.stack(frame['observation.sonic_state'])
            tokens = np.stack(frame['action.motion_token'])
            targets = np.stack(frame['action.dex1_target_position'])
            harness.curr_traj_data = frame
            harness.trajectory_lengths = np.array([len(frame)])
            # Out-of-order reads include tail padding and overlapping windows.
            for start in [len(frame)-1, len(frame)//2, 0, max(0, len(frame)//2-10)]:
                indices = np.minimum(np.arange(start, start+50), len(frame)-1)
                for enabled, expected_grips in [(False, targets), (True, labels)]:
                    harness.dex1_binary_gripper = enabled
                    action = np.concatenate([harness.get_state_or_action(episode, 'action', 'action.'+key, start)
                                             for key in ('motion_token', 'left_gripper', 'right_gripper')], axis=1)
                    np.testing.assert_array_equal(action[:, :64], tokens[indices])
                    np.testing.assert_array_equal(action[:, 64:], expected_grips[indices])
                    state = np.concatenate([harness.get_state_or_action(episode, 'state', 'state.'+key, start)
                                            for key in ('body', 'left_gripper', 'right_gripper')], axis=1)
                    np.testing.assert_array_equal(state, body[start:start+1])
                windows += 1
        values = np.concatenate(all_labels)
        stats = label_statistics(root)
        for side, key in enumerate(('left_gripper', 'right_gripper')):
            self.assertEqual(stats[key]['min'], [0.])
            self.assertEqual(stats[key]['max'], [1.])
            self.assertAlmostEqual(stats[key]['mean'][0], float(values[:, side].mean()))
        self.assertEqual(before, {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in tracked})
        print(f'Checked {len(paths)} episodes, {len(values)} frames, {windows} windows; closed fractions={values.mean(0).tolist()}')


if __name__ == '__main__':
    unittest.main()
