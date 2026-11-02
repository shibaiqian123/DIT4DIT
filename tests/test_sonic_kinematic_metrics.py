import unittest
import numpy as np
import torch
from DiT4DiT.evaluation.sonic_data import split_episodes,validation_windows
from DiT4DiT.evaluation.sonic_metrics import KinematicMetrics
from DiT4DiT.evaluation.sonic_kinematics import SonicKinematicDecoder


class FakeDecoder:
    def decode(self,x):
        x=torch.as_tensor(x,dtype=torch.float32)
        pos=x[...,:3,None].transpose(-1,-2).expand(*x.shape[:-1],2,3).clone()
        a=x[...,3]; c=a.cos(); s=a.sin(); z=a*0; o=z+1
        r=torch.stack((c,-s,z,s,c,z,z,z,o),-1).reshape(*a.shape,3,3)
        return pos,r[...,None,:,:].expand(*a.shape,2,3,3)


def example(key='episode:0',t=50):
    return dict(action=np.zeros((t,66),np.float32),action_mask=np.ones((t,66),bool),
                valid_time_mask=np.ones(t,bool),sample_id=key)


def statistics():
    return dict(action_contract='sonic_dex1_v1',normalization_mode='min_max',binary_indices=[],min=[-1.]*66,max=[1.]*66)


class Tests(unittest.TestCase):
    def test_split(self):
        s=split_episodes(range(52)); self.assertEqual(s,split_episodes(range(52)))
        self.assertEqual(len(s['val']),6); self.assertFalse(set(s['train'])&set(s['val']))
        w=validation_windows({1:3,2:6},5)
        self.assertEqual(len(w),5); self.assertEqual(len(set(w)),5)
        self.assertEqual(len(validation_windows({1:3,2:6},0)),9)

    def test_units_mask_groups(self):
        ex=example(); p=np.zeros((1,50,66),np.float32); p[0,:,0]=.01
        p[0,:,3]=np.pi/6
        ex['valid_time_mask'][-1]=False; p[0,-1,0]=1
        m=KinematicMetrics(FakeDecoder(),statistics());m.update(p,[ex]);v=m.compute()
        self.assertAlmostEqual(v['val/eef_kin_left_mean_cm'],1,places=5)
        self.assertAlmostEqual(v['val/eef_kin_right_rot_deg'],30,places=4)
        self.assertEqual(v['val/eef_kin_valid_frames'],49)
        self.assertEqual(len([k for k in v if '_cm_h' in k]),10)
        m.update(p,[ex]);self.assertEqual(m.compute()['val/eval_windows'],1)

    def test_p95_and_element_mask(self):
        ex=example();p=np.zeros((1,50,66),np.float32);p[0,:,0]=np.arange(50)/100
        ex['action_mask'][0,0]=False
        m=KinematicMetrics(FakeDecoder(),statistics());m.update(p,[ex]);v=m.compute()
        self.assertAlmostEqual(v['val/eef_kin_left_p95_cm'],float(np.quantile(np.arange(1,50),.95)),places=4)
        self.assertEqual(v['val/eef_kin_valid_frames'],49)
        self.assertAlmostEqual(v['val/token_mse'],float((p[0,:,:64]**2)[ex['action_mask'][:,:64]].mean()))

    def test_identity_empty_and_nonfinite(self):
        ex=example();m=KinematicMetrics(FakeDecoder(),statistics());m.update(np.zeros((1,50,66)),[ex])
        self.assertEqual(m.compute()['val/eef_kin_left_rot_deg'],0)
        ex['valid_time_mask'][:]=False;m=KinematicMetrics(FakeDecoder(),statistics());m.update(np.zeros((1,50,66)),[ex])
        with self.assertRaises(ValueError):m.compute()
        with self.assertRaises(ValueError):m.update(np.full((1,50,66),np.nan),[ex])

    def test_layout(self):
        # Known producer layout: all qpos followed by all qvel, then split into 58D rows.
        q=torch.arange(290.).reshape(10,29); velocity=10000+q
        output=torch.cat((torch.cat((q.flatten(),velocity.flatten())).reshape(10,58),torch.zeros(10,6)),-1)
        m=SonicKinematicDecoder.__new__(SonicKinematicDecoder);torch.nn.Module.__init__(m)
        m.frames=10; m.num_dofs=29; m.token_dim=32; m.max_num_tokens=2; m.token_total_dim=64
        m.output_frame_dim=64; m.decoder_input_features=('token',)
        m.decoder_output_feature_dims={'command_multi_future_nonflat':58,'motion_anchor_ori_heading_mf_nonflat':6}
        m.module=torch.nn.Sequential(torch.nn.Linear(64,640))
        with torch.no_grad():
            m.module[0].weight.zero_();m.module[0].bias.copy_(output.flatten())
        torch.testing.assert_close(m.decode_joints(torch.zeros(1,2,64)),q.expand(1,2,10,29))

if __name__=='__main__':unittest.main()
