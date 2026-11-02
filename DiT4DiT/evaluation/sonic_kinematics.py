"""Frozen SONIC v1.1 kinematic decoder; no simulator or policy dependencies."""
import ast
import hashlib
import pickle
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import torch
from torch import nn
from omegaconf import OmegaConf

OFFICIAL_SHA256 = 'af24831ae59424a0cf92cb56e9bb6dc1a59ab859fd055ba13187e9e6f0a59f43'
PREFIX = 'actor_module.decoders.g1_kin.module.'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_checkpoint(path):
    # Pin before allowing the inert trainer-state placeholder. Never import TRL.
    digest = sha256(path)
    if digest != OFFICIAL_SHA256:
        raise ValueError('Expected the pinned official SONIC v1.1 checkpoint')
    class Metadata:
        def __init__(self, *args, **kwargs):
            pass
        def __setstate__(self, state):
            self.state = state
    class Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module.startswith(('trl.', 'transformers.', 'accelerate.')):
                return type(name, (Metadata,), {'__module__': module})
            return super().find_class(module, name)
    class PinnedPickle:
        pass
    PinnedPickle.Unpickler = Unpickler
    PinnedPickle.load = staticmethod(pickle.load)
    ckpt = torch.load(path, map_location='cpu', weights_only=False,
                      pickle_module=PinnedPickle, mmap=True)
    return ckpt, digest


def official_body_order(source):
    tree = ast.parse(Path(source).read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'G1_ISAACLAB_JOINTS' for t in node.targets):
            names = ast.literal_eval(node.value)
            if len(names) == 30 and names[0] == 'pelvis':
                return names[1:]
    raise ValueError('Official G1 IsaacLab body order missing')


def quat_matrix(q):
    q = np.asarray(q, dtype=float)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]


class G1ForwardKinematics(nn.Module):
    """MJCF tree FK in pelvis coordinates, with IsaacLab-ordered input joints."""
    def __init__(self, xml_path, body_order):
        super().__init__()
        tree = ET.parse(xml_path).getroot()
        if tree.find('compiler').get('angle', 'degree') != 'radian':
            raise ValueError('Expected radian MJCF')
        root = tree.find("./worldbody/body[@name='pelvis']")
        if root is None:
            raise ValueError('Missing pelvis')
        self.nodes = []
        positions, rotations, axes, anchors = [], [], [], []
        seen = []
        def visit(body, parent):
            idx = len(self.nodes)
            name = body.attrib['name']
            joints = body.findall('joint')
            if len(joints) > 1:
                raise ValueError('Only one hinge per body supported')
            joint = joints[0] if joints and parent >= 0 else None
            slot = -1
            axis, anchor = [0, 0, 1], [0, 0, 0]
            if joint is not None:
                if joint.get('type', 'hinge') != 'hinge' or float(joint.get('ref', '0')) != 0:
                    raise ValueError('Unsupported joint convention')
                slot = body_order.index(name)
                seen.append(name)
                axis = list(map(float, joint.get('axis', '0 0 1').split()))
                anchor = list(map(float, joint.get('pos', '0 0 0').split()))
            if any(k in body.attrib for k in ('euler','axisangle','xyaxes','zaxis')):
                raise ValueError('Only quaternion body orientations supported')
            pos = list(map(float, body.get('pos', '0 0 0').split()))
            quat = list(map(float, body.get('quat', '1 0 0 0').split()))
            if parent < 0:
                pos, quat = [0, 0, 0], [1, 0, 0, 0]
            positions.append(pos); rotations.append(quat_matrix(quat))
            axes.append(axis); anchors.append(anchor)
            self.nodes.append((name, parent, slot))
            for child in body.findall('body'):
                visit(child, idx)
        visit(root, -1)
        if set(seen) != set(body_order) or len(seen) != 29:
            raise ValueError('Skeleton and official DOF order differ')
        self.wrists = [next(i for i,n in enumerate(self.nodes) if n[0] == side+'_wrist_yaw_link') for side in ('left','right')]
        for key, value in [('positions',positions),('rotations',rotations),('axes',axes),('anchors',anchors)]:
            self.register_buffer(key, torch.tensor(value, dtype=torch.float32))
        self.axes /= self.axes.norm(dim=-1, keepdim=True)

    def forward(self, q):
        if q.shape[-1] != 29 or not torch.isfinite(q).all():
            raise ValueError('Expected finite IsaacLab joint positions [...,29]')
        shape = q.shape[:-1]
        eye = torch.eye(3, device=q.device, dtype=q.dtype).expand(*shape, 3, 3)
        p, r = [], []
        for i, (_, parent, slot) in enumerate(self.nodes):
            local_p = self.positions[i].expand(*shape, 3)
            local_r = self.rotations[i].expand(*shape, 3, 3)
            if slot >= 0:
                axis = self.axes[i]
                x,y,z = axis.unbind()
                zero = x*0
                k = torch.stack((zero,-z,y,z,zero,-x,-y,x,zero)).reshape(3,3)
                angle = q[...,slot,None,None]
                rotation = eye + angle.sin()*k + (1-angle.cos())*(k@k)
                anchor = self.anchors[i]
                local_p = local_p + (local_r @ (anchor - (rotation @ anchor[...,None]).squeeze(-1))[...,None]).squeeze(-1)
                local_r = local_r @ rotation
            if parent < 0:
                p.append(local_p); r.append(local_r)
            else:
                p.append(p[parent]+(r[parent]@local_p[...,None]).squeeze(-1))
                r.append(r[parent]@local_r)
        return torch.stack([p[i] for i in self.wrists],-2), torch.stack([r[i] for i in self.wrists],-3)


class SonicKinematicDecoder(nn.Module):
    def __init__(self, checkpoint, sonic_root, device='cpu'):
        super().__init__()
        checkpoint, sonic_root = Path(checkpoint), Path(sonic_root)
        config_path = checkpoint.parent/'config.yaml'
        cfg = OmegaConf.load(config_path)
        backbone = cfg.algo.config.actor.backbone
        dec = backbone.decoders.g1_kin
        layer = dec.params.module_config_dict.layer_config
        frames = int(cfg.manager_env.commands.motion.num_future_frames)
        if (list(dec.inputs) != ['token'] or list(dec.outputs) != ['command_multi_future_nonflat','motion_anchor_ori_heading_mf_nonflat']
            or layer.type != 'MLP' or layer.activation != 'SiLU' or list(layer.hidden_dims) != [2048,1024,512,512]
            or int(backbone.max_num_tokens) != 2 or int(backbone.num_fsq_levels) != 32 or frames != 10):
            raise ValueError('Unsupported SONIC v1.1 configuration')
        self.frames = frames
        ckpt, digest = load_checkpoint(checkpoint)
        state = {k[len(PREFIX):]:v.float() for k,v in ckpt['policy_state_dict'].items() if k.startswith(PREFIX)}
        dims = [64,2048,1024,512,512,640]
        modules = []
        for i,(a,b) in enumerate(zip(dims[:-1],dims[1:])):
            modules.append(nn.Linear(a,b))
            if i < len(dims)-2: modules.append(nn.SiLU())
        self.module = nn.Sequential(*modules)
        self.module.load_state_dict(state, strict=True)
        del ckpt, state
        asset = cfg.manager_env.commands.motion.motion_lib_cfg.asset.assetFileName
        if asset != 'g1_29dof_rev_1_0.xml':
            raise ValueError('Unexpected training skeleton')
        skeleton = sonic_root/'gear_sonic/data/assets/robot_description/mjcf'/asset
        order_path = sonic_root/'gear_sonic/envs/manager_env/robots/g1.py'
        self.fk = G1ForwardKinematics(skeleton, official_body_order(order_path))
        self.metadata = dict(checkpoint=str(checkpoint), checkpoint_sha256=digest,
            config_sha256=sha256(config_path), skeleton=str(skeleton), skeleton_sha256=sha256(skeleton),
            joint_order_sha256=sha256(order_path), reference_frame='pelvis', eef_reference='wrist_yaw_link',
            decoder_frame_index=0, decoder_frame_offset_seconds=0., decoder_frames=frames)
        self.to(device=device, dtype=torch.float32).requires_grad_(False).eval()

    def decode_joints(self, tokens):
        if tokens.shape[-1] != 64 or not torch.isfinite(tokens).all():
            raise ValueError('Expected finite raw SONIC tokens [...,64]')
        # Output features are interleaved per decoder frame (58 command + 6 heading).
        output = self.module(tokens.float()).reshape(*tokens.shape[:-1], self.frames,64)
        command = output[...,:58].reshape(*tokens.shape[:-1], self.frames*58)
        # Producer concatenates ALL positions, then ALL velocities before reshaping.
        return command[...,:self.frames*29].reshape(*tokens.shape[:-1],self.frames,29)

    @torch.inference_mode()
    def decode(self, tokens):
        tokens = torch.as_tensor(tokens,device=self.module[0].weight.device,dtype=torch.float32)
        if tokens.ndim != 3:
            raise ValueError('decode expects [B,T,64]')
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            return self.fk(self.decode_joints(tokens)[...,0,:])
