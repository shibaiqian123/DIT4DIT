"""Real checkpoint/BaseModule parity and independent MuJoCo FK reference."""
import argparse
import ast
from types import SimpleNamespace
import importlib.util
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf
from DiT4DiT.evaluation.sonic_kinematics import SonicKinematicDecoder,official_body_order

p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--sonic-root',required=True);p.add_argument('--output',required=True)
p.add_argument('--dataset',type=Path)
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
# Execute the actual official methods without importing the full training stack.
source=root/'gear_sonic/trl/modules/universal_token_modules.py'
cls=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='UniversalTokenModule')
methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ('assemble_all_tokens','decode')]
ns={'torch':torch};exec(compile(ast.Module(body=methods,type_ignores=[]),str(source),'exec'),ns)
harness=SimpleNamespace(decoders={'g1_kin':reference},decoder_input_features={'g1_kin':['token']},
    decoder_output_feature_dims={'g1_kin':m.decoder_output_feature_dims},decoder_cond_features={},decoder_mask_features={})
assembled=ns['assemble_all_tokens'](harness,{'g1':tokens.reshape(12,2,32)}, {'g1':None},3,4)
torch.testing.assert_close(m.to_internal_tokens(tokens),assembled,atol=0,rtol=0)
fields=ns['decode'](harness,'g1_kin',{'token':assembled})
ours=m.decode_fields(tokens)
assert list(ours)==list(fields)
for key in fields:torch.testing.assert_close(ours[key],fields[key],atol=1e-6,rtol=1e-6)
field_errors={k:float((ours[k]-fields[k]).abs().max()) for k in fields}
source_rows=0
if a.dataset is not None:
    import pyarrow.parquet as pq
    report=json.loads((a.dataset/'meta/conversion_report.json').read_text())
    source_root=Path(report['source'])
    src_info=json.loads((source_root/'meta/info.json').read_text())
    out_info=json.loads((a.dataset/'meta/info.json').read_text())
    for entry in report['accepted']:
        def read(root,info,index):
            file=root/info['data_path'].format(episode_chunk=index//info['chunks_size'],episode_index=index)
            return np.asarray(pq.read_table(file,columns=['action.motion_token'])['action.motion_token'].to_pylist())
        original=read(source_root,src_info,entry['source_episode'])
        exported=read(a.dataset,out_info,entry['output_episode'])
        np.testing.assert_array_equal(original,exported)
        raw=torch.from_numpy(exported[:6].astype(np.float32)).reshape(2,3,64)
        internal=m.to_internal_tokens(raw)
        torch.testing.assert_close(internal.flatten(-2),raw,atol=0,rtol=0)
        ref_internal=ns['assemble_all_tokens'](harness,{'g1':raw.reshape(6,2,32)}, {'g1':None},2,3)
        expected_fields=ns['decode'](harness,'g1_kin',{'token':ref_internal})
        observed_fields=m.decode_fields(raw)
        for key in expected_fields:
            torch.testing.assert_close(observed_fields[key],expected_fields[key],atol=1e-6,rtol=1e-6)
        source_rows+=len(exported)

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
# Selected alternative links and fixed TCP transforms against independent
# MuJoCo body transforms, using homogeneous matrix composition as oracle.
from DiT4DiT.evaluation.sonic_kinematics import G1ForwardKinematics
from DiT4DiT.evaluation.sonic_reconstruction import EndEffectorTransform
from scipy.spatial.transform import Rotation
links = {'left':'left_wrist_roll_link', 'right':'right_wrist_roll_link'}
calibration = {side:{'translation_m':[.07, -.02, .03],
                    'quaternion_wxyz':[2**-.5, 0., 2**-.5, 0.]} for side in links}
transform = EndEffectorTransform(links,calibration)
fk_selected = G1ForwardKinematics(m.metadata['skeleton'],order,transform.links)
p_tcp,r_tcp = transform(*fk_selected(q))
tcp_error = 0.
for j,angles in enumerate(q):
    data.qpos[:]=0; data.qpos[3]=1
    for i,name in enumerate(order):
        body=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,name)
        data.qpos[model.jnt_qposadr[model.body_jntadr[body]]]=angles[i].item()
    mujoco.mj_forward(model,data)
    for side,name in enumerate(transform.links):
        bid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,name)
        world=np.eye(4); world[:3,:3]=data.xmat[bid].reshape(3,3); world[:3,3]=data.xpos[bid]
        offset=np.eye(4); offset[:3,3]=[.07,-.02,.03]
        offset[:3,:3]=Rotation.from_quat([0.,2**-.5,0.,2**-.5]).as_matrix()
        expected_tcp=world@offset
        tcp_error=max(tcp_error,float(np.max(np.abs(expected_tcp[:3,3]-p_tcp[j,side].numpy()))),
                      float(np.max(np.abs(expected_tcp[:3,:3]-r_tcp[j,side].numpy()))))
assert tcp_error < 2e-6,tcp_error
reconstructed=m.decode_reconstruction(tokens)
torch.testing.assert_close(reconstructed.joint_positions,m.decode_joints(tokens),atol=0,rtol=0)
for a_default,b_default in zip(m.decode(tokens),m.fk(reconstructed.joint_positions[...,0,:])):
    torch.testing.assert_close(a_default,b_default,atol=0,rtol=0)
try:
    G1ForwardKinematics(m.metadata['skeleton'],order,('missing_left','missing_right'))
except ValueError:
    pass
else:
    raise AssertionError('Missing links must fail explicitly')
result=dict(selected_link_tcp_max_abs=tcp_error,status='CHECKPOINT_BASEMODULE_AND_FK_PARITY_OK',mlp_max_abs=float((actual-expected).abs().max()),fk_position_max_abs_m=max_p,fk_rotation_max_abs=max_r,metadata=m.metadata,official_decode_max_abs=field_errors,source_export_rows_equal=source_rows)
Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
