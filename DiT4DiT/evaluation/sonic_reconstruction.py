"""Matched v1.1 reconstruction layout and explicitly calibrated link offsets."""
from dataclasses import dataclass

import torch
from torch import nn

COMMAND = 'command_multi_future_nonflat'
HEADING = 'motion_anchor_ori_heading_mf_nonflat'
DEFAULT_LINKS = {'left': 'left_wrist_yaw_link', 'right': 'right_wrist_yaw_link'}


@dataclass(frozen=True)
class KinematicReconstruction:
    joint_positions: torch.Tensor  # [B,H,F,29], IsaacLab order, radians
    joint_velocities: torch.Tensor  # [B,H,F,29], radians/second
    heading_first_two_columns: torch.Tensor  # [B,H,F,6], NOT a pelvis root pose


def parse_reconstruction(fields, frames=10, num_dofs=29):
    """Undo the official producer's position-then-velocity temporal flattening."""
    if set(fields) != {COMMAND, HEADING}:
        raise ValueError('Expected the matched v1.1 command and heading fields')
    command, heading = fields[COMMAND], fields[HEADING]
    if (command.ndim != 4 or command.shape[-2:] != (frames, 2*num_dofs)
            or heading.shape != (*command.shape[:2], frames, 6)):
        raise ValueError('Invalid v1.1 reconstruction field shapes')
    if any(not x.is_floating_point() or not torch.isfinite(x).all() for x in (command, heading)):
        raise ValueError('Reconstruction fields must be finite floating point tensors')
    pos, vel = command.flatten(-2).split(frames*num_dofs, dim=-1)
    shape = (*command.shape[:2], frames, num_dofs)
    return KinematicReconstruction(pos.reshape(shape), vel.reshape(shape), heading)


class EndEffectorTransform(nn.Module):
    """T_pelvis_tcp = T_pelvis_link @ T_link_tcp; left, right order.

    translation_m is TCP origin expressed in the selected link frame.
    quaternion_wxyz rotates TCP axes into that link frame. No calibration default.
    """
    def __init__(self, links=None, link_to_tcp=None):
        super().__init__()
        links = DEFAULT_LINKS if links is None else dict(links)
        if set(links) != {'left', 'right'}:
            raise ValueError('eef_links requires exactly left and right')
        self.links = tuple(links[s] for s in ('left', 'right'))
        if any(not isinstance(n, str) or not n.startswith(s+'_') for s,n in zip(('left','right'), self.links)):
            raise ValueError('Each end effector must select a link on its named side')
        self.enabled = link_to_tcp is not None
        if self.enabled and set(link_to_tcp) != {'left', 'right'}:
            raise ValueError('link_to_tcp requires explicit left and right transforms')
        translations, rotations, calibration = [], [], {}
        for side in ('left', 'right'):
            entry = link_to_tcp[side] if self.enabled else {'translation_m': [0,0,0], 'quaternion_wxyz': [1,0,0,0]}
            if set(entry) != {'translation_m', 'quaternion_wxyz'}:
                raise ValueError('Transform requires translation_m and quaternion_wxyz')
            t = torch.tensor(list(entry['translation_m']), dtype=torch.float32)
            q = torch.tensor(list(entry['quaternion_wxyz']), dtype=torch.float32)
            if t.shape != (3,) or q.shape != (4,) or not torch.isfinite(t).all() or not torch.isfinite(q).all():
                raise ValueError('Invalid fixed transform shape or nonfinite value')
            if abs(q.norm().item()-1) > 1e-4:
                raise ValueError('quaternion_wxyz must be unit length')
            q = q/q.norm()
            w,x,y,z = q.unbind()
            r = torch.stack((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
                             2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
                             2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y))).reshape(3,3)
            translations.append(t); rotations.append(r)
            calibration[side] = {'translation_m': t.tolist(), 'quaternion_wxyz': q.tolist()}
        self.register_buffer('translation', torch.stack(translations))
        self.register_buffer('rotation', torch.stack(rotations))
        self.metadata = {'eef_links': dict(links), 'link_to_tcp': calibration if self.enabled else None,
                         'eef_reference': 'tcp' if self.enabled else ('wrist_yaw_link' if dict(links)==DEFAULT_LINKS else 'link_origin')}

    def forward(self, position, rotation):
        if not self.enabled:
            return position, rotation
        return position + (rotation @ self.translation[...,None]).squeeze(-1), rotation @ self.rotation
