"""Validate converted data with native DiT4DiT loaders; no model or GPU required.

From the DiT4DiT repository: python examples/Real_G1/validate_sonic_dex1.py
The loader can create its normal step-index cache in the converted dataset only.
"""
import argparse
import ast
import copy
import json
import tempfile
from pathlib import Path
from typing import Dict

import numpy as np
import pyarrow.parquet as pq
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from DiT4DiT.dataloader.lerobot_datasets import make_LeRobotSingleDataset,get_vla_dataset,collate_fn
from DiT4DiT.dataloader.gr00t_lerobot.sonic_normalization import mark_statistics,unnormalize_sonic_actions


def validate(config_path):
    cfg=OmegaConf.load(config_path).datasets.vla_data
    root=Path(cfg.data_root_dir)/'block_into_cup_v1_1_dex1_dit4dit'
    report=json.loads((root/'meta/conversion_report.json').read_text())
    single=make_LeRobotSingleDataset(Path(cfg.data_root_dir),root.name,'g1_sonic_dex1',data_cfg=cfg)
    assert len(single.trajectory_ids)==len(report['accepted'])
    checked=0;max_error=0.
    for trajectory_id,length in zip(single.trajectory_ids,single.trajectory_lengths):
        table=pq.read_table(root/f'data/chunk-{int(trajectory_id)//1000:03d}/episode_{int(trajectory_id):06d}.parquet')
        action=np.concatenate([np.stack(table[k].to_pylist()) for k in ('action.motion_token','action.dex1_target_position')],axis=1)
        state=np.stack(table['observation.sonic_state'].to_pylist())
        for step in (0,int(length)//2,int(length)-1):
            raw=single.get_step_data(int(trajectory_id),step)
            expected=action[np.minimum(np.arange(step,step+50),int(length)-1)]
            observed=np.concatenate([raw[k] for k in single.modality_keys['action']],axis=1)
            np.testing.assert_array_equal(observed,expected)
            np.testing.assert_array_equal(np.concatenate([raw[k] for k in single.modality_keys['state']],axis=1),state[step:step+1])
            transformed=single.transforms(copy.deepcopy(raw))
            normalized=np.concatenate([transformed[k] for k in single.modality_keys['action']],axis=1)
            states=np.concatenate([transformed[k] for k in single.modality_keys['state']],axis=1)
            assert normalized.shape==(50,66) and states.shape==(1,31)
            assert np.isfinite(normalized).all() and np.isfinite(states).all()
            assert abs(normalized).max()<=1.00001 and abs(states).max()<=1.00001
            keys=['motion_token','left_gripper','right_gripper']
            stats={stat:np.concatenate([np.asarray(single.metadata.statistics.action[k].model_dump()[stat]) for k in keys]).tolist()
                   for stat in ('min','max')}
            restored=unnormalize_sonic_actions(normalized,mark_statistics(stats))
            np.testing.assert_allclose(restored,expected,atol=2e-6,rtol=2e-6)
            max_error=max(max_error,float(abs(restored-expected).max()));checked+=1
            assert len(raw['video.ego_view'])==17
    mix=get_vla_dataset(cfg,mode='val')
    with tempfile.TemporaryDirectory() as tmp:
        stats_path=Path(tmp)/'statistics.json'
        mix.save_dataset_statistics(stats_path)
        statistics=json.loads(stats_path.read_text())
        action_stats=statistics['new_embodiment']['action']
        assert action_stats['normalization_mode']=='min_max' and action_stats['binary_indices']==[]
        assert len(action_stats['mask'])==66 and all(action_stats['mask'])
        # Execute the exact static method body without loading VLM training modules.
        tree=ast.parse(Path('DiT4DiT/model/framework/base_framework.py').read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='baseframework')
        fn=copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='unnormalize_actions'))
        fn.decorator_list=[]
        ns={'np':np,'Dict':Dict}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'base_framework.py','exec'),ns)
        normalized=np.linspace(-.9,.9,66)[None,:]
        np.testing.assert_array_equal(ns['unnormalize_actions'](normalized,action_stats),unnormalize_sonic_actions(normalized,action_stats))
        constant=copy.deepcopy(action_stats);constant['min'][6]=constant['max'][6]=.375
        assert ns['unnormalize_actions'](normalized,constant)[0,6]==.375
        # Existing representations retain their original binary-index-6 behavior.
        legacy={'q01':[0.]*7,'q99':[1.]*7}
        assert ns['unnormalize_actions'](np.zeros((1,7)),legacy)[0,6]==.5
    # Exercise mixture processing once for every episode, including tail padding.
    original=mix.sample_step
    for tid,length in zip(single.trajectory_ids,single.trajectory_lengths):
        mix.sample_step=lambda index,t=int(tid),s=int(length)-1:(mix.datasets[0],t,s)
        sample=mix[0]
        assert sample['action'].shape==(50,66) and sample['state'].shape==(1,31)
        assert len(sample['image'])==9 and all(tuple(im.shape)==(3,224,224) for im in sample['image'])
        assert sample['action_mask'].all() and sample['state_mask'].all()
        assert np.isfinite(sample['action']).all() and np.isfinite(sample['state']).all()
        assert sample['lang']
    mix.sample_step=original
    batch=next(iter(DataLoader(mix,batch_size=2,num_workers=0,collate_fn=collate_fn)))
    assert len(batch)==2 and all(x['action'].shape==(50,66) for x in batch)
    result={'episodes':len(single.trajectory_ids),'frames':report['frames'],'checked_single_windows':checked,
            'action_shape':[50,66],'state_shape':[1,31],'batch_size':len(batch),
            'images_per_sample':9,'max_action_roundtrip_error':max_error,'status':'passed'}
    (root/'meta/loader_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='DiT4DiT/config/real_robot/dit4dit_sonic_dex1.yaml')
    validate(p.parse_args().config)
