import unittest
from types import SimpleNamespace
import numpy as np
import torch
from rtc_sampler import soft_mask, guided_velocity, RTCActionHead
from protocol import ConsumptionClock, shifted

class RTCMathTests(unittest.TestCase):
    def test_mask_exact_and_padding(self):
        m=soft_mask(50,10,25)
        self.assertTrue(torch.equal(m[:10],torch.ones(10)))
        self.assertEqual(float(m[25:].sum()),0)
        c=torch.tensor(15/16)
        self.assertAlmostEqual(float(m[10]),float(c*torch.expm1(c)/torch.expm1(torch.tensor(1.))),6)
        for d,s in [(26,26),(-1,25),(10,5),(10,41)]:
            with self.assertRaises(ValueError):soft_mask(50,d,s)
        shifted(np.ones((50,66)),25,10)
        with self.assertRaises(ValueError):shifted(np.ones((50,66)),49,10)

    def test_analytic_vjp_direction(self):
        x=torch.ones(1,4,2);prev=torch.zeros_like(x);t=.5
        v=guided_velocity(lambda x,t:2*x,x,t,prev,torch.ones(4),.9)
        # x0 == 0, hence zero correction
        torch.testing.assert_close(v,2*x)
        v=guided_velocity(lambda x,t:.5*x,x,t,prev,torch.ones(4),.9)
        # J=0.75 I; error=-0.75 x; scaled correction=-1.35 x
        torch.testing.assert_close(v,1.85*x)
        self.assertLess(float((x-.1*v).norm()),float((x-.1*.5*x).norm()))

    def test_no_guidance_and_inference_guard(self):
        x=torch.randn(1,4,2)
        for mask,t,alpha in [(torch.zeros(4),.5,.9),(torch.ones(4),1,.9),(torch.ones(4),.5,0)]:
            torch.testing.assert_close(guided_velocity(lambda x,t:x*2,x,t,x,mask,alpha),x*2)
        with torch.inference_mode(),self.assertRaises(RuntimeError):
            guided_velocity(lambda x,t:x,x,.5,x,torch.ones(4),.9)

    def test_adapter_zero_guidance_exact_euler_parity(self):
        class Encoder(torch.nn.Module):
            def forward(self,x,t):return x*.25
        class Core(torch.nn.Module):
            def forward(self,hidden_states,**kwargs):return hidden_states
        h=torch.nn.Module()
        h.config=SimpleNamespace(action_horizon=50,action_dim=66,add_pos_embed=False)
        h.action_horizon=50;h.num_timestep_buckets=1000;h.num_inference_timesteps=4
        h.state_encoder=None;h.action_encoder=Encoder();h.model=Core();h.action_decoder=torch.nn.Identity()
        noise=torch.randn(1,50,66);expected=noise.clone()
        for _ in range(4):expected=expected-.25*(expected*.25)
        out=RTCActionHead(h).predict_action_rtc(torch.zeros(1,1,2),None,torch.zeros_like(noise),
                                              10,25,guidance_alpha=0,initial_noise=noise)
        torch.testing.assert_close(out,expected,atol=0,rtol=0)

    def test_frozen_parameters_and_no_graph_retention(self):
        layer=torch.nn.Linear(2,2).requires_grad_(False)
        x=torch.randn(1,4,2)
        for _ in range(30):
            out=guided_velocity(lambda x,t:layer(x),x,.5,torch.zeros_like(x),torch.ones(4),.9)
            self.assertFalse(out.requires_grad)
        self.assertTrue(all(p.grad is None for p in layer.parameters()))

class ClockTests(unittest.TestCase):
    def reply(self,c,r,id='b'):
        return dict(ok=True,session=c.session,request_id=r['request_id'],chunk_id=id,
                    normalized=np.zeros((50,66)).tolist(),window_offset=r['consumed'])
    def test_replan_and_skip_delay_no_duplicates(self):
        c=ConsumptionClock('a');r=c.request('first');c.accept(self.reply(c,r))
        self.assertEqual([c.consume() for _ in range(25)],list(range(25)))
        r=c.request('next')
        self.assertEqual([c.consume() for _ in range(8)],list(range(25,33)))
        c.accept(self.reply(c,r,'c'))
        self.assertEqual(c.consume(),8)
        self.assertEqual(list(c.delays),[10,8])
    def test_exhaustion_is_terminal(self):
        c=ConsumptionClock('a');r=c.request('first');c.accept(self.reply(c,r))
        for _ in range(25):c.consume()
        r=c.request('slow')
        for _ in range(25):c.consume()
        with self.assertRaises(ValueError):c.consume()
        with self.assertRaises(ValueError):c.accept(self.reply(c,r,'late'))
    def test_reconnect_and_concurrent_requests(self):
        c=ConsumptionClock('new');r=c.request('first')
        old=self.reply(c,r);old['session']='old'
        self.assertFalse(c.accept(old))
        with self.assertRaises(ValueError):c.request('second')
        c.accept(self.reply(c,r))
        with self.assertRaises(ValueError):c.request('early')
    def test_wrong_window_is_rejected(self):
        c=ConsumptionClock('a');r=c.request('first');reply=self.reply(c,r)
        reply['window_offset']=1
        with self.assertRaises(ValueError):c.accept(reply)

    def test_returned_block_expired(self):
        c=ConsumptionClock('a');r=c.request('first');c.total=50
        with self.assertRaises(ValueError):c.accept(self.reply(c,r))

if __name__=='__main__':unittest.main()
