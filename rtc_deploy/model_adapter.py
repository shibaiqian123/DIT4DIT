"""DiT4DiT resource loader. Imported only on the inference server."""
import base64
import hashlib
import io
import os
import json
import time
from pathlib import Path
import numpy as np
from latency_trace import mark, stamp
from PIL import Image
import torch
from rtc_sampler import RTCActionHead
from protocol import block


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda:f.read(4*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def install_prompt_cache(extractor):
    """Single-entry deployment cache. Never cache trainable/training encoders.

    Instance lifetime ties the cache to the loaded checkpoint. Disable at startup
    with RTC_DISABLE_PROMPT_CACHE=1 for same-noise comparisons.
    """
    original = extractor._get_prompt_embeds
    cached_key = cached_value = None

    def get_prompt_embeds(prompt, *, device, dtype, max_sequence_length):
        nonlocal cached_key, cached_value
        encoder = extractor.text_encoder
        if encoder.training or any(p.requires_grad for p in encoder.parameters()):
            cached_key = cached_value = None
            return original(prompt, device=device, dtype=dtype,
                            max_sequence_length=max_sequence_length)
        prompts = (prompt,) if isinstance(prompt, str) else tuple(prompt)
        key = (prompts, str(device), dtype, max_sequence_length)
        if key != cached_key:
            with torch.no_grad():
                value = original(prompt, device=device, dtype=dtype,
                                 max_sequence_length=max_sequence_length)
            cached_value = value.detach()
            cached_key = key
        return cached_value

    extractor._get_prompt_embeds = get_prompt_embeds


class Policy:
    def __init__(self, run, checkpoint, cosmos, seed=42):
        from omegaconf import OmegaConf
        from DiT4DiT.model.framework import build_framework
        from DiT4DiT.model.framework.DiT4DiT import DiT4DiT
        from DiT4DiT.model.framework.share_tools import dict_to_namespace
        from DiT4DiT.dataloader.gr00t_lerobot.sonic_normalization import unnormalize_sonic_actions
        run = Path(run).resolve()
        ckpt = run/'checkpoints'/checkpoint
        cfg = OmegaConf.load(run/'config.yaml')
        if (cfg.framework.action_model.action_dim, cfg.framework.action_model.state_dim,
                cfg.framework.action_model.action_horizon) != (66,31,50):
            raise ValueError('Expected trained SONIC 31-state / 50x66 contract')
        self.stats = json.loads((run/'dataset_statistics.json').read_text())['new_embodiment']
        self.unnormalize = unnormalize_sonic_actions
        self.unnormalize(np.zeros((50,66)), self.stats['action'])
        cfg.framework.cosmos25.base_model = str(Path(cosmos).resolve())
        cfg.trainer.pretrained_checkpoint = None
        torch.manual_seed(seed)
        self.model = build_framework(dict_to_namespace(OmegaConf.to_container(cfg, resolve=True)))
        sd = torch.load(ckpt, map_location='cpu', weights_only=True, mmap=True)
        self.model.load_state_dict(sd, strict=True)
        del sd
        self.model.eval().requires_grad_(False).to(device='cuda', dtype=torch.bfloat16)
        self.prompt_cache_enabled = os.environ.get('RTC_DISABLE_PROMPT_CACHE') != '1'
        if self.prompt_cache_enabled:
            install_prompt_cache(self.model.backbone_interface.extractor)
        self.feature_early_exit = os.environ.get('RTC_FEATURE_EARLY_EXIT') == '1'
        extractor = self.model.backbone_interface.extractor
        if self.feature_early_exit and not hasattr(extractor, '_transformer_feature_call'):
            raise RuntimeError('Update Cosmos25.py before enabling feature early exit')
        extractor.feature_early_exit = self.feature_early_exit
        self.rtc = RTCActionHead(self.model.action_model)
        self.info = dict(checkpoint=str(ckpt), checkpoint_sha256=sha(ckpt),
                         config_sha256=sha(run/'config.yaml'), statistics_sha256=sha(run/'dataset_statistics.json'),
                         cosmos=str(Path(cosmos).resolve()), action_shape=[50,66], token_shape=[50,64],
                         cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES', ''),
                         state_dim=31, inference_steps=self.model.action_model.num_inference_timesteps,
                         rtc_mode='test_time', guidance_alpha=0.9, seed=seed,
                         feature_early_exit=self.feature_early_exit, prompt_cache_enabled=self.prompt_cache_enabled)

    def example(self, obs):
        state = np.asarray(obs['state'], dtype=np.float32)
        if state.shape != (31,) or not np.isfinite(state).all():
            raise ValueError('Expected finite raw state[31]')
        if not isinstance(obs['lang'], str) or not obs['lang'].strip():
            raise ValueError('Task instruction required')
        if not np.isfinite(float(obs['observed_at'])) or not isinstance(obs['observation_id'], str):
            raise ValueError('Observation identity/timestamp required')
        encoding = obs.get('image_encoding', 'png')
        if encoding == 'rgb8_base64':
            if obs.get('image_shape') != [224, 224, 3]:
                raise ValueError('Expected RGB uint8 shape [224,224,3]')
            wire = obs['image_rgb']
            if not isinstance(wire, str) or len(wire) != 200704:
                raise ValueError('Invalid RGB payload length')
            raw = base64.b64decode(wire, validate=True)
            if len(raw) != 224*224*3:
                raise ValueError('Invalid RGB byte count')
            image = Image.fromarray(np.frombuffer(raw, dtype=np.uint8).reshape(224,224,3))
        elif encoding == 'png':
            image = Image.open(io.BytesIO(base64.b64decode(obs['image_png'], validate=True))).convert('RGB')
        else:
            raise ValueError('Unsupported image encoding')
        if image.size != (224,224):
            image = image.resize((224,224), Image.Resampling.BILINEAR)
        lo, hi = [np.asarray(self.stats['state'][key]) for key in ('min','max')]
        if lo.shape != (31,) or hi.shape != (31,) or not np.isfinite([lo,hi]).all():
            raise ValueError('Invalid state statistics')
        norm = np.zeros((1,31), dtype=np.float32)
        mask = hi != lo
        norm[:, mask] = 2*(state[mask]-lo[mask])/(hi[mask]-lo[mask])-1
        return dict(image=[image], lang=obs['lang'], state=norm)

    def predict(self, obs, previous=None, delay=0, consumed=0, alpha=0.9):
        trace_id = obs.get('_latency_request', obs['observation_id'])
        mark(trace_id, 'predict_start')
        ex = self.example(obs)
        torch.cuda.synchronize()
        start = time.monotonic()
        mark(trace_id, 'preprocess_done')
        if previous is None:
            with torch.no_grad():
                norm = self.model.predict_action([ex])['normalized_actions'][0]
        else:
            backbone = self.model.backbone_interface
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                inputs = backbone.build_cosmos_inputs(images=[ex['image']], instructions=[ex['lang']])
                features = backbone(**inputs, output_attentions=False, output_hidden_states=True,
                                    return_dict=True).hidden_states[-1].detach()
            state = torch.from_numpy(np.array([ex['state']])).to(features)
            # Match framework precision, but enable gradients only inside RTC action steps.
            with torch.autocast('cuda', dtype=torch.float32):
                out = self.rtc.predict_action_rtc(features, state, previous[None], delay, consumed, alpha)
            norm = out.float().cpu().numpy()[0]
        norm = block(norm)
        actions = self.unnormalize(norm, self.stats['action'])
        if not np.isfinite(actions).all():
            raise ValueError('Nonfinite denormalized actions')
        torch.cuda.synchronize()
        mark(trace_id, 'predict_done')
        return dict(normalized=norm.tolist(), motion_tokens=actions[:,:64].tolist(),
                    gripper_actions=actions[:,64:].tolist(), inference_seconds=time.monotonic()-start,
                    out_of_range_fraction=float((np.abs(norm)>1).mean()))
