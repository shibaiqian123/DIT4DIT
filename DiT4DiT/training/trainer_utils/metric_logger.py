"""Optional training metric backends; only the global main process writes."""
import json
from numbers import Real
from pathlib import Path


class MetricLogger:
    def __init__(self, config, is_main_process):
        self.writer = None
        self.wandb_run = None
        self.jsonl = None
        selected = getattr(config, "trackers", ["wandb"])
        if isinstance(selected, str):
            selected = [selected]
        selected = set(selected or [])
        unknown = selected - {"tensorboard", "wandb", "jsonl"}
        if unknown:
            raise ValueError(f"Unsupported trackers: {sorted(unknown)}")
        if not is_main_process:
            return
        root = Path(config.output_dir)
        root.mkdir(parents=True, exist_ok=True)
        try:
            if "tensorboard" in selected:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(str(root / "tensorboard"), flush_secs=10)
            if "jsonl" in selected:
                self.jsonl = (root / "metrics.jsonl").open("a", encoding="utf-8")
            if "wandb" in selected:
                import wandb
                (root / "wandb").mkdir(exist_ok=True)
                self.wandb_run = wandb.init(
                    name=config.run_id, dir=str(root / "wandb"),
                    project=config.wandb_project, entity=config.wandb_entity,
                    group="vla-train",
                )
        except BaseException:
            self.close()
            raise

    def log(self, metrics, step):
        scalars = {}
        for key, value in metrics.items():
            if hasattr(value, "numel"):
                if value.numel() != 1:
                    continue
                value = value.detach().item()
            elif hasattr(value, "item"):
                if getattr(value, "size", 1) != 1:
                    continue
                value = value.item()
            if isinstance(value, Real):
                scalars[key] = float(value)
        if self.writer is not None:
            for key, value in scalars.items():
                self.writer.add_scalar(key, value, global_step=step)
            self.writer.flush()
        if self.jsonl is not None:
            self.jsonl.write(json.dumps({"step": step, "metrics": scalars}) + "\n")
            self.jsonl.flush()
        if self.wandb_run is not None:
            self.wandb_run.log(metrics, step=step)

    def close(self):
        # Clear handles first so repeated cleanup is harmless.
        writer, self.writer = self.writer, None
        jsonl, self.jsonl = self.jsonl, None
        run, self.wandb_run = self.wandb_run, None
        try:
            if writer is not None:
                writer.close()
        finally:
            try:
                if jsonl is not None:
                    jsonl.close()
            finally:
                if run is not None:
                    run.finish()
