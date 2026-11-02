"""Exercise actual framework/trainer methods without loading the video backbone."""
import ast
import contextlib
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from omegaconf import OmegaConf
from test_sonic_training_loss import Decoder
from test_sonic_kinematic_metrics import statistics
from DiT4DiT.evaluation.sonic_training_loss import SonicEndEffectorLoss

ROOT=Path(__file__).resolve().parents[1]

def method(path,cls,name):
    tree=ast.parse((ROOT/path).read_text())
    node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls)
    fn=next(n for n in node.body if isinstance(n,ast.FunctionDef) and n.name==name)
    ns={'torch':torch,'np':np,'List':list,'Tuple':tuple}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),ns)
    return ns[name]

class WiringTests(unittest.TestCase):
    def test_framework_repeated_reference_and_padding_mask(self):
        class Backbone:
            def build_cosmos_inputs(self,**kw): return {}
            def __call__(self,**kw): return SimpleNamespace(hidden_states=[torch.zeros(1,2,4)])
        class Head:
            def __call__(self,features,labels,mask,state,return_clean_estimate=False):
                self.received=return_clean_estimate
                return (torch.tensor(1.),labels) if return_clean_estimate else torch.tensor(1.)
        head=Head()
        cfg=OmegaConf.create({'trainer':{'repeated_diffusion_steps':2,'eef_training_loss':{'enabled':True}}})
        fake=SimpleNamespace(backbone_interface=Backbone(),video_fm_only=False,config=cfg,
                             future_action_window_size=49,action_model=head)
        ex={'image':[], 'lang':'test','action':np.zeros((50,66)), 'action_mask':np.ones((50,66),bool),
            'valid_time_mask':np.ones(50,bool),'decoder_reference_mask':np.ones((50,1),bool)}
        ex['valid_time_mask'][-1]=False;ex['decoder_reference_mask'][0]=False
        forward=method('DiT4DiT/model/framework/DiT4DiT.py','DiT4DiT','forward')
        out=forward(fake,[ex])
        self.assertTrue(head.received)
        self.assertEqual(out['eef_valid_elements'].shape,(2,50,66))
        self.assertFalse(out['eef_valid_elements'][:,[0,49]].any())
        self.assertTrue(out['eef_valid_elements'][:,1:49].all())
        cfg.trainer.eef_training_loss.enabled=False
        self.assertEqual(set(forward(fake,[ex])),{'action_loss'})
        self.assertFalse(head.received)

    def test_trainer_adds_auxiliary_loss_and_backpropagates(self):
        class Model(torch.nn.Module):
            def __init__(self): super().__init__();self.p=torch.nn.Parameter(torch.full((1,2,66),.2))
            def forward(self,batch):
                return {'action_loss':self.p.sum()*0,'eef_clean_actions':self.p,
                        'eef_target_actions':torch.zeros_like(self.p),'eef_valid_elements':torch.ones_like(self.p,dtype=torch.bool)}
        model=Model()
        accelerator=SimpleNamespace(accumulate=lambda _:contextlib.nullcontext(),backward=lambda x:x.backward(),sync_gradients=False)
        cfg=OmegaConf.create({'trainer':{'eef_training_loss':{'enabled':True}}})
        fake=SimpleNamespace(model=model,accelerator=accelerator,config=cfg,kinematic_validation_dataset=[1],
                             _eef_training_criterion=SonicEndEffectorLoss(Decoder(),statistics()))
        step=method('DiT4DiT/training/train.py','VLATrainer','_train_step')
        logs=step(fake,None)
        self.assertGreater(logs['train/eef_weighted'],0)
        self.assertEqual(logs['total_loss'],logs['train/eef_weighted'])
        self.assertGreater(model.p.grad[...,:64].abs().sum().item(),0)
        self.assertEqual(model.p.grad[...,64:].abs().sum().item(),0)

if __name__=='__main__':unittest.main()
