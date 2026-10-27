import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
from omegaconf import OmegaConf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from DiT4DiT.training.trainer_utils.config_tracker import wrap_config
from DiT4DiT.training.trainer_utils.metric_logger import MetricLogger


class MetricLoggerTests(unittest.TestCase):
    def test_real_events_and_jsonl_with_wrapped_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = wrap_config(OmegaConf.create({"trackers": ["tensorboard", "jsonl"], "output_dir": tmp}))
            # TensorBoard-only logging must not initialize or require W&B.
            with patch.dict("sys.modules", {"wandb": None}):
                log = MetricLogger(cfg, True)
                log.log({"loss": torch.tensor(1.25), "learning_rate/base": np.float64(0.001),
                         "data_time": 0.2, "epoch": 1, "ignored": "text",
                         "vector": torch.ones(2)}, 7)
                log.close()
                log.close()
            events = EventAccumulator(str(Path(tmp) / "tensorboard")).Reload()
            self.assertEqual(set(events.Tags()["scalars"]), {"loss", "learning_rate/base", "data_time", "epoch"})
            event = events.Scalars("loss")[0]
            self.assertEqual((event.step, event.value), (7, 1.25))
            row = json.loads((Path(tmp) / "metrics.jsonl").read_text())
            self.assertEqual(row["step"], 7)
            self.assertEqual(row["metrics"]["loss"], 1.25)

    def test_worker_does_not_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "worker"
            log = MetricLogger(SimpleNamespace(trackers=["tensorboard", "jsonl", "wandb"], output_dir=str(root)), False)
            log.log({"loss": 2.}, 1)
            log.close()
            self.assertFalse(root.exists())

    def test_wandb_can_coexist_and_default_is_preserved(self):
        for trackers in (None, ["wandb", "tensorboard", "jsonl"]):
            with self.subTest(trackers=trackers), tempfile.TemporaryDirectory() as tmp:
                cfg = SimpleNamespace(output_dir=tmp, run_id="test", wandb_project="test", wandb_entity="test")
                if trackers is not None:
                    cfg.trackers = trackers
                fake = Mock()
                with patch.dict("sys.modules", {"wandb": fake}):
                    log = MetricLogger(cfg, True)
                    log.log({"loss": 2.}, 3)
                    log.close()
                    log.close()
                fake.init.return_value.log.assert_called_once_with({"loss": 2.}, step=3)
                fake.init.return_value.finish.assert_called_once()

    def test_disabled_and_unknown(self):
        log = MetricLogger(SimpleNamespace(trackers=[] ,output_dir="unused"), False)
        log.close()
        with self.assertRaises(ValueError):
            MetricLogger(SimpleNamespace(trackers=["typo"]), True)

    def test_initialization_failure_closes_writer(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = SimpleNamespace(trackers=["tensorboard", "wandb"], output_dir=tmp,
                                  run_id="test", wandb_project="test", wandb_entity="test")
            fake = Mock()
            fake.init.side_effect = RuntimeError("network failure")
            with patch("torch.utils.tensorboard.SummaryWriter") as writer, patch.dict("sys.modules", {"wandb": fake}):
                with self.assertRaises(RuntimeError):
                    MetricLogger(cfg, True)
                writer.return_value.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
