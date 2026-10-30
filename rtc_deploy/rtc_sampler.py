"""Psi0 test-time RTC VJP guidance; no learned parameters or robot imports.
Reference/license: THIRD_PARTY.md. All RTC tensors are normalized actions.
"""
import math
import torch


def soft_mask(h, d, s, *, device=None):
    if any(type(x) is not int for x in (h, d, s)) or not (h > 0 and 0 <= d <= s <= h-d):
        raise ValueError('RTC requires integer 0 <= d <= s <= H-d')
    mask = torch.zeros(h, device=device, dtype=torch.float32)
    mask[:d] = 1
    end = h-s
    if end > d:
        c = (end-torch.arange(d, end, device=device, dtype=torch.float32))/(end-d+1)
        mask[d:end] = c*torch.expm1(c)/math.expm1(1)
    return mask


def guided_velocity(velocity_fn, x, t, previous, mask, alpha):
    """VJP of x0 wrt x; do not differentiate the error or accumulate parameter grads."""
    if torch.is_inference_mode_enabled():
        raise RuntimeError('RTC cannot run inside torch.inference_mode()')
    if not math.isfinite(alpha) or alpha < 0 or not 0 < t <= 1:
        raise ValueError('Invalid guidance alpha or noise time')
    if alpha == 0 or not bool(mask.any()) or t == 1:
        with torch.no_grad():
            return velocity_fn(x, t).detach()
    with torch.enable_grad():
        sample = x.detach().requires_grad_(True)
        v = velocity_fn(sample, t)
        x0 = sample-t*v
        error = (previous-x0.detach()).float()*mask[None, :, None]
        correction = torch.autograd.grad(x0, sample, grad_outputs=error.to(x0.dtype),
                                         retain_graph=False, create_graph=False)[0].float()
    norm = correction.norm()
    if not torch.isfinite(norm) or not torch.isfinite(error).all():
        raise ValueError('Nonfinite RTC correction')
    if norm >= 1e-8:
        correction = alpha*error.norm()/(t*norm)*correction
    else:
        correction = torch.zeros_like(correction)
    return v.detach()-correction.to(v.dtype)


class RTCActionHead:
    """Deployment adapter for an existing FlowmatchingActionHead; state_dict unchanged."""
    def __init__(self, head):
        self.head = head

    def predict_action_rtc(self, vl_embs, state, previous, inference_delay,
                           execution_horizon, guidance_alpha=0.9, initial_noise=None):
        h = self.head
        if torch.is_inference_mode_enabled():
            raise RuntimeError('RTC requires no_grad, not inference_mode')
        if any(p.requires_grad for p in h.parameters()):
            raise ValueError('Freeze action head parameters before RTC')
        shape = (1, h.config.action_horizon, h.config.action_dim)
        if vl_embs.shape[0] != 1 or h.action_horizon != shape[1]:
            raise ValueError('RTC supports batch one and matching horizon configuration')
        previous = torch.as_tensor(previous, device=vl_embs.device, dtype=torch.float32)
        if tuple(previous.shape) != shape or not torch.isfinite(previous).all():
            raise ValueError('Invalid normalized previous action block')
        mask = soft_mask(shape[1], inference_delay, execution_horizon, device=vl_embs.device)
        features = vl_embs.detach()
        with torch.no_grad():
            sf = h.state_encoder(state.detach()) if state is not None else None
            actions = (torch.randn(shape, device=features.device, dtype=features.dtype)
                       if initial_noise is None else initial_noise.detach().clone().to(features))
        if tuple(actions.shape) != shape or not torch.isfinite(actions).all():
            raise ValueError('Invalid initial noise')
        def velocity(x, t):
            ts = torch.full((1,), int(t*h.num_timestep_buckets), device=x.device)
            af = h.action_encoder(x, ts)
            if h.config.add_pos_embed:
                af = af+h.position_embedding(torch.arange(x.shape[1], device=x.device))[None]
            sa = torch.cat((sf, af), dim=1) if sf is not None else af
            output = h.model(hidden_states=sa, encoder_hidden_states=features, timestep=ts)
            return h.action_decoder(output)[:, -h.action_horizon:]
        steps = h.num_inference_timesteps
        if type(steps) is not int or steps <= 0:
            raise ValueError('Invalid checkpoint inference steps')
        for i in range(steps):
            t = 1-i/float(steps)
            v = guided_velocity(velocity, actions, t, previous, mask, guidance_alpha)
            actions = (actions-v/steps).detach()
            if not torch.isfinite(actions).all():
                raise ValueError('Nonfinite sampled actions')
        return actions
