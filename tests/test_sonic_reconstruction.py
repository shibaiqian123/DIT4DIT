import unittest
import torch
from DiT4DiT.evaluation.sonic_reconstruction import (
    COMMAND, HEADING, parse_reconstruction, EndEffectorTransform)


class ReconstructionTests(unittest.TestCase):
    def test_temporal_layout(self):
        q = torch.arange(2*3*10*29).float().reshape(2,3,10,29)
        v = -q-10000
        fields = {COMMAND: torch.cat((q.flatten(-2),v.flatten(-2)),-1).reshape(2,3,10,58),
                  HEADING: torch.ones(2,3,10,6)}
        result = parse_reconstruction(fields)
        torch.testing.assert_close(result.joint_positions,q,rtol=0,atol=0)
        torch.testing.assert_close(result.joint_velocities,v,rtol=0,atol=0)
        # Official loss's first-half-rows extraction must recover the same qpos.
        torch.testing.assert_close(fields[COMMAND][...,:5,:].reshape_as(q),q)
        fields[HEADING] = torch.zeros(2,3,10,6)  # heading unused in pelvis FK
        torch.testing.assert_close(parse_reconstruction(fields).joint_positions,q)
        fields[COMMAND][0,0,0,0] = float('nan')
        with self.assertRaises(ValueError): parse_reconstruction(fields)
        with self.assertRaises(ValueError): parse_reconstruction({COMMAND:q})

    def test_fixed_transform_composition(self):
        transform = {s:{'translation_m':[.1,0,0], 'quaternion_wxyz':[2**-.5,0,0,2**-.5]} for s in ('left','right')}
        module = EndEffectorTransform(link_to_tcp=transform)
        p = torch.zeros(2,3,2,3)
        r = torch.tensor([[0.,-1,0],[1,0,0],[0,0,1]]).expand(2,3,2,3,3)
        pos,rot = module(p,r)
        torch.testing.assert_close(pos,torch.tensor([0.,.1,0]).expand_as(pos))
        torch.testing.assert_close(rot,torch.diag(torch.tensor([-1.,-1,1])).expand_as(rot),atol=3e-7,rtol=0)
        unchanged = EndEffectorTransform()(p,r)
        self.assertIs(unchanged[0],p); self.assertIs(unchanged[1],r)

    def test_invalid_configuration(self):
        with self.assertRaises(ValueError): EndEffectorTransform({'left':'right_wrist_yaw_link','right':'left_wrist_yaw_link'})
        with self.assertRaises(ValueError): EndEffectorTransform(link_to_tcp={'left':{}})
        for q in ([0,0,0,0],[2,0,0,0],[float('nan'),0,0,0]):
            with self.assertRaises(ValueError):
                EndEffectorTransform(link_to_tcp={s:{'translation_m':[0,0,0],'quaternion_wxyz':q} for s in ('left','right')})

if __name__ == '__main__': unittest.main()
