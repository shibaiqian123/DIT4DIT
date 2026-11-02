"""Real checkpoint/BaseModule parity and independent MuJoCo FK reference."""
import argparse
import importlib.util
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf
from DiT4DiT.evaluation.sonic_kinematics import SonicKinematicDecoder,official_body_order

p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--sonic-root',required=True);p.add_argument('--output',required=True)
a=p.parse_args();root=Path(a.sonic_root)
m=SonicKinematicDecoder(a.checkpoint,root)
spec=importlib.util.spec_from_file_location('official_base',root/'gear_sonic/trl/modules/base_module.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
reference=module.BaseModule(input_dim=32,output_dim=64,num_input_temporal_dims=2,num_output_temporal_dims=10,
    module_config_dict=OmegaConf.create({'layer_config':{'type':'MLP','hidden_dims':[2048,1024,512,512],'activation':'SiLU'}}))
reference.module.load_state_dict(m.module.state_dict(),strict=True);reference.eval()
torch.manual_seed(42);tokens=torch.randn(3,4,64)
with torch.no_grad():
    expected=reference(tokens.reshape(3,4,2,32)).reshape(3,4,640)
    actual=m.module(tokens)
torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-6)
import mujoco
xml=ET.parse(m.metadata['skeleton']).getroot()
# Remove rendering/dynamics-only elements; retain the complete kinematic body/joint tree.
for node in list(xml):
    if node.tag not in ('compiler','worldbody','default'):xml.remove(node)
for parent in xml.iter():
    for child in list(parent):
        if child.tag in ('geom','site','camera','light'):parent.remove(child)
model=mujoco.MjModel.from_xml_string(ET.tostring(xml,encoding='unicode'));data=mujoco.MjData(model)
order=official_body_order(root/'gear_sonic/envs/manager_env/robots/g1.py')
q=torch.cat([torch.zeros(1,29),torch.randn(8,29)*.3])
pos,rot=m.fk(q)
max_p=max_r=0.
for j,angles in enumerate(q):
    data.qpos[:]=0;data.qpos[3]=1
    for i,name in enumerate(order):
        body=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,name)
        joint=model.body_jntadr[body];data.qpos[model.jnt_qposadr[joint]]=angles[i].item()
    mujoco.mj_forward(model,data)
    for side,name in enumerate(('left_wrist_yaw_link','right_wrist_yaw_link')):
        bid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,name)
        max_p=max(max_p,float(np.max(np.abs(data.xpos[bid]-pos[j,side].numpy()))))
        max_r=max(max_r,float(np.max(np.abs(data.xmat[bid].reshape(3,3)-rot[j,side].numpy()))))
assert max_p<2e-6 and max_r<2e-6,(max_p,max_r)
result=dict(status='CHECKPOINT_BASEMODULE_AND_FK_PARITY_OK',mlp_max_abs=float((actual-expected).abs().max()),fk_position_max_abs_m=max_p,fk_rotation_max_abs=max_r,metadata=m.metadata)
Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
