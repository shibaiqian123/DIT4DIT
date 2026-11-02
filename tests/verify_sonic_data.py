"""Actual dataset split, train-only extrema, padding and saved-stat round trip."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from DiT4DiT.evaluation.sonic_data import build_datasets,build_train_loader
from DiT4DiT.evaluation.sonic_kinematics import sha256
from DiT4DiT.training.trainer_utils.config_tracker import AccessTrackedConfig

cfg=OmegaConf.load('DiT4DiT/config/real_robot/dit4dit_sonic_dex1.yaml')
cfg.output_dir='results/sonic_kinematic_validation/data_check'
cfg.trainer.kinematic_validation.max_windows=4
train,val=build_datasets(cfg,cfg.output_dir)
ds=train.datasets[0];manifest=val.validation_manifest
original_hash=sha256(ds.dataset_path/'meta/stats_gr00t.json')
assert set(e for e,s in ds.all_steps)==set(manifest['train'])
assert not set(manifest['train'])&set(manifest['val'])
for i in range(100):assert int(train.sample_step(i)[1]) in manifest['train']
# Cross-check action token extrema directly from only the selected episode parquet files.
arrays=[]
for e in manifest['train']:
    path=ds.dataset_path/ds.data_path_pattern.format(episode_chunk=e//ds.chunk_size,episode_index=e)
    arrays.append(np.stack(pd.read_parquet(path,columns=['action.motion_token'])['action.motion_token']))
x=np.concatenate(arrays)
st=ds.metadata.statistics.action['motion_token']
np.testing.assert_allclose(st.min,x.min(0));np.testing.assert_allclose(st.max,x.max(0))
tail=val[-1];assert tail['valid_time_mask'].sum()==1
assert tail['decoder_reference_mask'].shape==(50,1) and tail['decoder_reference_mask'].sum()==1
stats_path=Path(cfg.output_dir)/'dataset_statistics.json';train.save_dataset_statistics(stats_path)
_,restored=build_datasets(cfg,Path(cfg.output_dir)/'restored',old_statistics=stats_path)
np.testing.assert_array_equal(val[0]['action'],restored[0]['action'])
assert sha256(ds.dataset_path/'meta/stats_gr00t.json')==original_hash
loader=build_train_loader(AccessTrackedConfig(cfg))
sample=next(iter(loader))[0]
assert sample['action'].shape==(50,66) and sample['state'].shape==(1,31)
assert sample['decoder_reference_mask'].shape==(50,1)
np.testing.assert_array_equal(sample['decoder_reference_mask'][:,0],sample['valid_time_mask'])
assert loader.kinematic_validation_dataset.validation_manifest==manifest
report=dict(status='REAL_DATA_SPLIT_STATS_PADDING_AND_TRAIN_LOADER_OK',train_episodes=len(manifest['train']),val_episodes=len(manifest['val']),tail_valid_frames=int(tail['valid_time_mask'].sum()),source_statistics_unchanged=True)
(Path(cfg.output_dir)/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
