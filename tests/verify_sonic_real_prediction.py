"""Run the offline CLI, then one action-head backward with captured real features.

Accepts the same arguments as evaluate_sonic. Does not step any optimizer.
"""
import json
import sys
from pathlib import Path
import numpy as np
import torch
from DiT4DiT.model.framework.DiT4DiT import DiT4DiT
from DiT4DiT.model.modules.action_model.ActionDiT import FlowmatchingActionHead
from DiT4DiT.evaluation import evaluate_sonic
from DiT4DiT.evaluation.sonic_kinematics import SonicKinematicDecoder
from DiT4DiT.evaluation.sonic_training_loss import SonicEndEffectorLoss
from omegaconf import OmegaConf

captured={}
original_policy=DiT4DiT.predict_action
original_head=FlowmatchingActionHead.predict_action

def capture_policy(self,examples,**kwargs):
    if 'examples' not in captured: captured['examples']=examples
    return original_policy(self,examples,**kwargs)

def capture_head(self,vl_embs,state=None):
    if 'head' not in captured:
        captured.update(head=self,features=vl_embs.detach().cpu(),state=None if state is None else state.detach().cpu())
    return original_head(self,vl_embs,state)

DiT4DiT.predict_action=capture_policy
FlowmatchingActionHead.predict_action=capture_head
try:
    evaluate_sonic.main()
finally:
    DiT4DiT.predict_action=original_policy
    FlowmatchingActionHead.predict_action=original_head

args=sys.argv
out=Path(args[args.index('--output')+1]);run=Path(args[args.index('--run')+1])
config=OmegaConf.load('DiT4DiT/config/real_robot/dit4dit_sonic_dex1.yaml').trainer
stats=json.loads((run/'dataset_statistics.json').read_text())['new_embodiment']['action']
head=captured['head'].float().requires_grad_(True).train()
device=head.device
features=captured['features'].to(device,dtype=torch.float32).clone()
state=None if captured['state'] is None else captured['state'].to(device,dtype=torch.float32).clone()
examples=captured['examples']
labels=torch.tensor(np.asarray([e['action'] for e in examples]),device=device,dtype=torch.float32)
mask=torch.tensor(np.asarray([e['action_mask'] for e in examples]),device=device,dtype=torch.bool)
times=torch.tensor(np.asarray([e['valid_time_mask'] for e in examples]),device=device,dtype=torch.bool)
mask=mask & times[...,None]
decoder=SonicKinematicDecoder(config.kinematic_validation.checkpoint,config.kinematic_validation.sonic_root,device=device)
criterion=SonicEndEffectorLoss(decoder,stats).to(device)
torch.manual_seed(42)
fm,clean=head(features,labels,mask,state,return_clean_estimate=True)
losses=criterion(clean,labels,mask)
parameter=head.action_decoder.layer2.weight
aux_grad,clean_grad=torch.autograd.grad(losses['weighted'],(parameter,clean),retain_graph=True)
assert torch.isfinite(aux_grad).all() and aux_grad.abs().sum()>0
assert torch.isfinite(clean_grad).all() and clean_grad[...,64:].abs().sum()==0
(fm+losses['weighted']).backward()
assert all(p.grad is None for p in decoder.parameters())
assert all(p.grad is None or torch.isfinite(p.grad).all() for p in head.parameters())
result=dict(status='REAL_FEATURE_ACTION_HEAD_EEF_BACKWARD_OK',optimizer_steps=0,
    flow_loss=float(fm.detach()),eef_losses={k:float(v.detach()) for k,v in losses.items()},
    eef_gradient_action_decoder_norm=float(aux_grad.norm()),
    decoder_parameter_gradients=0,conditioning='captured real policy backbone features',
    clean_action_formula='x_t - t * predicted_velocity',dex1_eef_gradient=0)
(out/'training_gradient_verification.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
