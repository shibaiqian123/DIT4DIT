import unittest
import numpy as np
import torch
from DiT4DiT.evaluation.sonic_training_loss import SonicEndEffectorLoss,clean_flow_estimate
from test_sonic_kinematic_metrics import statistics,FakeDecoder

class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__();self.scale=torch.nn.Parameter(torch.tensor(1.))
    def decode_differentiable(self,x):
        return FakeDecoder().decode(x*self.scale)

class TrainingTests(unittest.TestCase):
    def test_flow_parameterization(self):
        x=torch.randn(2,50,66);n=torch.randn_like(x);t=torch.tensor([.1,.9])[:,None,None]
        torch.testing.assert_close(clean_flow_estimate((1-t)*x+t*n,t,n-x),x)

    def test_gradient_freeze_padding_and_dex1(self):
        criterion=SonicEndEffectorLoss(Decoder(),statistics())
        prediction=torch.full((1,4,66),.2,requires_grad=True)
        target=torch.zeros_like(prediction);mask=torch.ones_like(prediction,dtype=torch.bool)
        mask[:,2:]=False
        losses=criterion(prediction,target,mask)
        losses['weighted'].backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertGreater(prediction.grad[:,:2,:64].abs().sum().item(),0)
        self.assertEqual(prediction.grad[:,2:].abs().sum().item(),0)
        self.assertEqual(prediction.grad[...,64:].abs().sum().item(),0)
        self.assertIsNone(criterion.decoder.scale.grad)
        self.assertFalse(criterion.decoder.scale.requires_grad)
        criterion.train();self.assertFalse(criterion.decoder.training)

    def test_identity_empty_clipping_and_inverse(self):
        criterion=SonicEndEffectorLoss(Decoder(),statistics())
        for valid in (True,False):
            x=torch.zeros(1,2,66,requires_grad=True)
            result=criterion(x,torch.zeros_like(x),torch.full_like(x,valid,dtype=torch.bool))
            result['weighted'].backward()
            self.assertEqual(result['weighted'].item(),0)
            self.assertTrue(torch.isfinite(x.grad).all())
        from DiT4DiT.dataloader.gr00t_lerobot.sonic_normalization import unnormalize_sonic_actions
        stats=statistics();stats['min']=list(np.arange(66)*.1);stats['max']=list(np.arange(66)*.2+1)
        criterion=SonicEndEffectorLoss(Decoder(),stats)
        x=torch.linspace(-2,2,132).reshape(1,2,66)
        np.testing.assert_allclose(criterion.raw_tokens(x).numpy(),unnormalize_sonic_actions(x.numpy(),stats)[...,:64],rtol=1e-6,atol=1e-6)

if __name__=='__main__':unittest.main()
