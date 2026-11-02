"""Offline held-out evaluation; never starts a service or robot interface.

python -m DiT4DiT.evaluation.evaluate_sonic --help
"""
import argparse
import json
from pathlib import Path
import torch
from omegaconf import OmegaConf
from .sonic_data import build_datasets
from .sonic_evaluator import SonicEvaluator


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--cosmos',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--evaluation-config',type=Path,default=Path('DiT4DiT/config/real_robot/dit4dit_sonic_dex1.yaml'))
    p.add_argument('--max-windows',type=int,default=256,help='0 = full validation split')
    p.add_argument('--step',type=int,default=0)
    p.add_argument('--cpu-offload-text-encoder',action='store_true',help='Reduce VRAM using Accelerate CPU offload; offline only')
    p.add_argument('--legacy-diagnostic',action='store_true',help='Explicitly allow a checkpoint trained without episode holdout')
    a=p.parse_args()
    cfg=OmegaConf.load(a.run/'config.yaml')
    cfg.trainer.kinematic_validation=OmegaConf.load(a.evaluation_config).trainer.kinematic_validation
    cfg.trainer.kinematic_validation.max_windows=a.max_windows
    cfg.datasets.vla_data.data_root_dir=str(a.data_root)
    cfg.framework.cosmos25.base_model=str(a.cosmos)
    cfg.trainer.pretrained_checkpoint=None
    if (cfg.framework.action_model.action_dim,cfg.framework.action_model.state_dim,cfg.framework.action_model.action_horizon)!=(66,31,50):
        raise ValueError('Expected SONIC action66/state31/horizon50 model')
    cfg.datasets.vla_data.max_action_dim=66
    cfg.datasets.vla_data.max_state_dim=31
    source_split=a.run/'validation_split.json'
    a.output.mkdir(parents=True,exist_ok=True)
    target_split=a.output/'validation_split.json'
    if source_split.exists():
        if target_split.exists() and target_split.read_bytes()!=source_split.read_bytes():
            raise ValueError('Output split differs from training split')
        if not target_split.exists(): target_split.write_bytes(source_split.read_bytes())
    elif not a.legacy_diagnostic:
        raise ValueError('Run has no independent split. Use --legacy-diagnostic only for diagnostic metrics.')
    _,dataset=build_datasets(cfg,a.output,old_statistics=a.run/'dataset_statistics.json',full=a.max_windows==0)
    dataset.statistics_provenance=('legacy_training_statistics_diagnostic' if not source_split.exists() else 'saved_training_split_and_statistics')
    from DiT4DiT.model.framework import build_framework
    from DiT4DiT.model.framework.share_tools import dict_to_namespace
    torch.manual_seed(int(cfg.trainer.kinematic_validation.seed))
    model=build_framework(dict_to_namespace(OmegaConf.to_container(cfg,resolve=True)))
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=True,mmap=True)
    model.load_state_dict(state,strict=True)
    del state
    model.to(device='cuda',dtype=torch.bfloat16).eval().requires_grad_(False)
    if a.cpu_offload_text_encoder:
        from accelerate import cpu_offload
        cpu_offload(model.backbone_interface.extractor.text_encoder,execution_device=torch.device('cuda:0'))
        torch.cuda.empty_cache()
    evaluator=SonicEvaluator(cfg.trainer.kinematic_validation,dataset,a.output,a.run/'dataset_statistics.json')
    evaluator.metadata.update(policy_checkpoint=str(a.checkpoint),policy_run=str(a.run),cpu_offload_text_encoder=a.cpu_offload_text_encoder)
    from torch.utils.tensorboard import SummaryWriter
    with SummaryWriter(str(a.output/'tensorboard')) as writer:
        metrics=evaluator.run(model,a.step,writer)
        for key,value in metrics.items(): writer.add_scalar(key,value,a.step)
        writer.flush()
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    events=EventAccumulator(str(a.output/'tensorboard')).Reload()
    for key in metrics:
        event=events.Scalars(key)[-1]
        if event.step!=a.step: raise AssertionError('TensorBoard step mismatch')
    image=events.Images('val/eef_kin_trajectory')[-1]
    if image.step!=a.step: raise AssertionError('TensorBoard trajectory step mismatch')
    result={'status':'REAL_PREDICTION_AND_TENSORBOARD_OK','step':a.step,'metrics':metrics,'metadata':evaluator.metadata}
    (a.output/'evaluation_result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
