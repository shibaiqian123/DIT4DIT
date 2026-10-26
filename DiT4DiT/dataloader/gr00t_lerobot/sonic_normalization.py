"""Explicit SONIC continuous action contract; no binary gripper channel."""
import numpy as np

CONTRACT = 'sonic_dex1_v1'


def mark_statistics(stats):
    stats.update(action_contract=CONTRACT,normalization_mode='min_max',binary_indices=[])
    stats['mask']=[True]*len(stats['min'])
    return stats


def unnormalize_sonic_actions(actions,stats):
    if stats.get('action_contract')!=CONTRACT or stats.get('normalization_mode')!='min_max':
        raise ValueError('Expected SONIC min_max action statistics')
    if stats.get('binary_indices')!=[]:raise ValueError('SONIC actions are all continuous')
    lo=np.asarray(stats['min']);hi=np.asarray(stats['max'])
    a=np.asarray(actions)
    if a.shape[-1]!=66 or lo.shape!=(66,) or hi.shape!=(66,):raise ValueError('SONIC action dimension must be 66')
    return (np.clip(a,-1,1)+1)*.5*(hi-lo)+lo
