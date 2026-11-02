"""Differentiable frozen-decoder supervision of clean flow action estimates."""
import torch
from torch import nn
from DiT4DiT.dataloader.gr00t_lerobot.sonic_normalization import unnormalize_sonic_actions


def clean_flow_estimate(noisy, time, predicted_velocity):
    # x_t = (1-t)*x_0 + t*noise, v = noise-x_0.
    return noisy.float() - time.float() * predicted_velocity.float()


class SonicEndEffectorLoss(nn.Module):
    def __init__(self, decoder, statistics, position_weight=1., rotation_weight=.1):
        super().__init__()
        # Validate using exactly the deployment normalization contract.
        import numpy as np
        unnormalize_sonic_actions(np.zeros((1,66)), statistics)
        self.decoder = decoder.requires_grad_(False).eval()
        self.register_buffer('lo', torch.tensor(statistics['min'][:64], dtype=torch.float32))
        self.register_buffer('span', torch.tensor(statistics['max'][:64], dtype=torch.float32)-self.lo)
        if not torch.isfinite(self.lo).all() or not torch.isfinite(self.span).all() or (self.span<0).any():
            raise ValueError('Invalid SONIC min-max ranges')
        self.position_weight, self.rotation_weight = float(position_weight), float(rotation_weight)
        if min(self.position_weight,self.rotation_weight)<0 or not all(np.isfinite([self.position_weight,self.rotation_weight])):
            raise ValueError('EEF loss weights must be finite and nonnegative')

    def train(self, mode=True):
        super().train(mode)
        self.decoder.eval()
        return self

    def raw_tokens(self, action):
        # Exactly one inverse transform. Clip matches deployment; saturated
        # components have zero gradient (no straight-through approximation).
        return (action[...,:64].float().clamp(-1,1)+1)*.5*self.span+self.lo

    def forward(self, prediction, target, valid_elements):
        if prediction.shape != target.shape or prediction.shape != valid_elements.shape or prediction.shape[-1]!=66:
            raise ValueError('EEF loss expects matching [B,H,66] actions and masks')
        valid = valid_elements[...,:64].bool().all(-1)
        if not valid.any():
            zero=prediction.float().sum()*0
            return {'position_m2':zero, 'rotation_rad2':zero, 'weighted':zero}
        if not torch.isfinite(prediction[valid]).all() or not torch.isfinite(target[valid]).all():
            raise ValueError('Nonfinite valid action in EEF training')
        # Select valid frames before decoding; invalid padded labels cannot leak.
        pred = self.raw_tokens(prediction[valid]).unsqueeze(0)
        label = self.raw_tokens(target[valid]).unsqueeze(0)
        self.decoder.eval()
        with torch.autocast(device_type=prediction.device.type, enabled=False):
            pp,pr=self.decoder.decode_differentiable(pred)
            with torch.no_grad():
                tp,tr=self.decoder.decode_differentiable(label)
            position=(pp-tp).square().sum(-1).mean()
            relative=pr.transpose(-1,-2)@tr
            vee=torch.stack((relative[...,2,1]-relative[...,1,2],relative[...,0,2]-relative[...,2,0],relative[...,1,0]-relative[...,0,1]),-1)
            sine=torch.linalg.vector_norm(vee,dim=-1)*.5
            cosine=((relative.diagonal(dim1=-2,dim2=-1).sum(-1)-1)*.5).clamp(-1,1)
            rotation=torch.atan2(sine,cosine).square().mean()
            weighted=self.position_weight*position+self.rotation_weight*rotation
        return {'position_m2':position,'rotation_rad2':rotation,'weighted':weighted}
