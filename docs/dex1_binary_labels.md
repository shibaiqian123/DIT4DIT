# Dex1 binary labels at load time

Enable in `datasets.vla_data` in your SONIC training configuration:

```yaml
dex1_binary_gripper: true
dex1_open_threshold: 0.2
dex1_close_threshold: 0.8
```

The option defaults to false, preserving existing continuous-target runs. It requires
`action_contract: sonic_dex1_v1`, `lerobot_version: v2.0`, and
`delete_pause_frame: false`. The v2 reader also handles this dataset's v2.1 layout.
Use a fresh run directory; do not reuse old continuous-target statistics caches.

The loader uses `teleop.dex1_trigger` and requires `dex1.trigger_valid == 1`
for both hands. Each complete episode is converted before action-window sampling:
trigger <= 0.2 means open (0), >= 0.8 means closed (1), otherwise retain the
previous state independently for each hand. A first frame in the deadband is
initialized using the midpoint (default 0.5; equal means closed). These thresholds
are a configurable initial labeling policy, not a validated grasp-event annotation.

Only `action.left_gripper` and `action.right_gripper` change. Token64, state31,
source Parquet and source metadata remain untouched. Last-frame action padding
repeats the last binary label. Invalid/missing triggers fail instead of falling
back to motor positions. In-memory grip statistics are computed from labels;
min/max are fixed to 0/1 even for a constant hand, so the existing min-max
transform produces -1/+1. Exported statistics record `dex1_label_encoding`;
`binary_indices` remains empty because training retains continuous regression.

This change implements data loading only. The model architecture, loss and
inference adapters are unchanged. A model trained with this option must not be
used with the existing motor-radian inference adapter until binary thresholding
and 0/1-to-radian conversion are implemented.

CPU verification (no model/GPU):

```bash
PYTHONPATH=. python tests/test_dex1_binary.py
DEX1_TEST_DATASET=/path/to/dataset PYTHONPATH=. python tests/test_dex1_binary.py
```

The real-data test executes the repository's raw-read/padding methods directly,
without importing video/VLM dependencies; it does not validate a full DataLoader
or training run.
