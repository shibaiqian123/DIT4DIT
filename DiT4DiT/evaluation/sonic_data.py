"""Episode-isolated SONIC datasets and immutable split/statistics manifests."""
import copy
import fcntl
import hashlib
import json
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader


def split_episodes(episodes, seed=42, fraction=.1):
    episodes=sorted(int(x) for x in episodes)
    if len(episodes)<2: raise ValueError('At least two episodes required for holdout')
    ids=np.random.default_rng(seed).permutation(episodes)
    count=max(1,min(len(ids)-1,int(np.ceil(len(ids)*fraction))))
    return dict(train=sorted(map(int,ids[count:])),val=sorted(map(int,ids[:count])))


def validation_windows(lengths, limit=256):
    # Round-robin evenly spaced frames covers episodes before adding depth.
    count=sum(lengths.values()) if not limit else min(int(limit),sum(lengths.values()))
    allocation={e:0 for e in lengths}
    while sum(allocation.values())<count:
        for e in sorted(lengths):
            if allocation[e]<lengths[e] and sum(allocation.values())<count: allocation[e]+=1
    per={e:list(np.linspace(0,lengths[e]-1,n,dtype=int)) for e,n in allocation.items() if n}
    return [(e,int(steps[k])) for k in range(max(map(len,per.values()))) for e,steps in per.items() if k<len(steps)]


def build_datasets(cfg, output_dir, *, old_statistics=None, full=False):
    from DiT4DiT.dataloader.lerobot_datasets import get_vla_dataset
    from DiT4DiT.dataloader.gr00t_lerobot.datasets import calculate_dataset_statistics
    from DiT4DiT.dataloader.gr00t_lerobot.schema import DatasetStatisticalValues
    data_cfg=cfg.datasets.vla_data
    if data_cfg.get('action_contract')!='sonic_dex1_v1' or data_cfg.get('dex1_binary_gripper',False):
        raise ValueError('Kinematic validation currently requires continuous SONIC token64 + Dex1(1+1)')
    if data_cfg.get('delete_pause_frame',False):
        raise ValueError('Kinematic evaluation requires original episode frame indices')
    if int(data_cfg.get('max_action_dim',32))!=66 or int(data_cfg.get('max_state_dim',64))!=31:
        raise ValueError('SONIC validation requires max_action_dim=66 and max_state_dim=31')
    if int(cfg.framework.action_model.action_horizon)!=50:
        raise ValueError('SONIC validation currently requires action_horizon=50')
    mixture=get_vla_dataset(data_cfg=data_cfg)
    if len(mixture.datasets)!=1:
        raise ValueError('Explicit per-dataset split required for multi-dataset mixtures')
    ds=mixture.datasets[0]
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    lengths={int(e):int(n) for e,n in zip(ds.trajectory_ids,ds.trajectory_lengths)}
    identity={'dataset':str(ds.dataset_path.resolve()),'lengths':{str(k):v for k,v in lengths.items()}}
    validation=cfg.trainer.kinematic_validation
    with (output_dir/'.kinematic_data.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        manifest_path=output_dir/'validation_split.json'
        if manifest_path.exists():
            manifest=json.loads(manifest_path.read_text())
            if manifest['identity']!=identity: raise ValueError('Dataset differs from saved split')
        else:
            manifest=dict(identity=identity, seed=int(validation.seed), fraction=float(validation.val_fraction),
                **split_episodes(lengths,int(validation.seed),float(validation.val_fraction)))
            manifest_path.write_text(json.dumps(manifest,indent=2))
        train_ids=set(manifest['train']); val_ids=set(manifest['val'])
        if train_ids & val_ids or train_ids | val_ids != set(lengths) or not train_ids or not val_ids:
            raise ValueError('Invalid persisted episode split')
        cache=output_dir/'kinematic_train_metadata.json'
        if old_statistics is None:
            if cache.exists():
                cached=json.loads(cache.read_text())
                if cached['split']!=manifest: raise ValueError('Cached statistics split differs')
                values=cached['statistics']
            else:
                paths=[ds.dataset_path/ds.data_path_pattern.format(episode_chunk=e//ds.chunk_size,episode_index=e) for e in sorted(train_ids)]
                raw=calculate_dataset_statistics(paths)
                values={}
                for modality in ('state','action'):
                    values[modality]={}
                    for key in ds.modality_keys[modality]:
                        meta=ds.lerobot_modality_meta.get_key_meta(key)
                        values[modality][key.split('.',1)[1]]={s:np.asarray(v)[meta.start:meta.end].tolist() for s,v in raw[meta.original_key].items()}
                cache.write_text(json.dumps({'split':manifest,'statistics':values}))
            for modality,groups in values.items():
                for key,stats in groups.items():
                    getattr(ds.metadata.statistics,modality)[key]=DatasetStatisticalValues.model_validate(stats)
            mixture.update_metadata({'percentile_mixing_method':'min_max'})
        else:
            saved=json.loads(Path(old_statistics).read_text())[ds.tag]
            if saved['action'].get('action_contract') != 'sonic_dex1_v1' or saved['action'].get('normalization_mode') != 'min_max':
                raise ValueError('Saved statistics are not SONIC min-max statistics')
            for modality in ('state','action'):
                offset=0
                for full_key in ds.modality_keys[modality]:
                    key=full_key.split('.',1)[1]
                    width=int(np.prod(getattr(ds.metadata.modalities,modality)[key].shape))
                    stats={k:list(saved[modality][k][offset:offset+width]) for k in ('mean','std','min','max','q01','q99')}
                    if any(len(v)!=width for v in stats.values()): raise ValueError('Saved statistic width mismatch')
                    getattr(ds.metadata.statistics,modality)[key]=DatasetStatisticalValues.model_validate(stats)
                    offset+=width
                if len(saved[modality]['min'])!=offset: raise ValueError('Saved statistic trailing dimensions')
            mixture.update_metadata({'percentile_mixing_method':'min_max'})
    # Copy before restricting train all_steps; transforms and metadata are independent.
    mixture._sonic_validity_metadata=True
    validation_ds=copy.deepcopy(mixture)
    vd=validation_ds.datasets[0]
    ds._all_steps=np.asarray([(e,s) for e,s in ds.all_steps if int(e) in train_ids])
    ds._trajectory_ids=np.asarray(sorted(train_ids)); ds._trajectory_lengths=np.asarray([lengths[e] for e in sorted(train_ids)])
    mixture._dataset_lengths=np.array([len(ds.all_steps)])
    mixture._trajectory_sampling_weights=[np.ones(len(train_ids))/len(train_ids)]
    vd._trajectory_ids=np.asarray(sorted(val_ids)); vd._trajectory_lengths=np.asarray([lengths[e] for e in sorted(val_ids)])
    windows=validation_windows({e:lengths[e] for e in sorted(val_ids)},0 if full else int(validation.max_windows))
    validation_ds._kinematic_windows=windows
    vd._all_steps=np.asarray(windows)
    validation_ds._dataset_lengths=np.array([len(windows)])
    validation_ds.mode='val'; vd.transforms.eval()
    validation_ds.validation_manifest=manifest
    validation_ds.statistics_provenance='legacy_training_statistics_diagnostic' if old_statistics else 'train_episodes_only'
    validation_ds.statistics_source=str(old_statistics) if old_statistics else str(cache)
    return mixture,validation_ds


def build_train_loader(cfg):
    from DiT4DiT.dataloader.lerobot_datasets import collate_fn
    import torch.distributed as dist
    old=cfg.trainer.kinematic_validation.get('legacy_statistics',None)
    if cfg.trainer.get('pretrained_checkpoint',None) and not (Path(cfg.output_dir)/'validation_split.json').exists():
        raise ValueError('A pretrained policy may already have seen the holdout; use the offline diagnostic entrypoint or a run with its original split')
    if cfg.trainer.get('is_resume',False) and not (Path(cfg.output_dir)/'kinematic_train_metadata.json').exists():
        raise ValueError('Resume requires the original training-only statistics manifest')
    if cfg.trainer.get('is_resume',False) and not (Path(cfg.output_dir)/'validation_split.json').exists():
        raise ValueError('Cannot retrofit an independent holdout into an already-trained run; use offline diagnostic evaluation')
    train,val=build_datasets(cfg,cfg.output_dir,old_statistics=old)
    if old:
        raise ValueError('Legacy statistics are supported only by the offline diagnostic entrypoint')
    if not dist.is_initialized() or dist.get_rank()==0:
        train.save_dataset_statistics(Path(cfg.output_dir)/'dataset_statistics.json')
    loader=DataLoader(train,batch_size=cfg.datasets.vla_data.per_device_batch_size,collate_fn=collate_fn,num_workers=4)
    loader.kinematic_validation_dataset=val
    return loader
