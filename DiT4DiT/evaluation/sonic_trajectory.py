"""Headless pelvis-local wrist trajectory comparison for TensorBoard and PNG."""
import numpy as np


def trajectory_figure(row, sample_id):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,3,figsize=(12,6),squeeze=False)
    valid=np.asarray(row['valid'],bool)
    for side in range(2):
        for axis in range(3):
            ax=axes[side,axis]
            for field,label in (('target_position','label'),('prediction_position','prediction')):
                values=np.asarray(row[field])[:,side,axis]*100
                ax.plot(np.where(valid,values,np.nan),label=label)
            ax.set_title(('left','right')[side]+' '+('x','y','z')[axis])
            ax.set_xlabel('action horizon index');ax.set_ylabel('pelvis-local position (cm)')
            ax.grid(alpha=.25)
    axes[0,0].legend()
    fig.suptitle(f'{sample_id} | SONIC reference frame 0')
    fig.tight_layout()
    return fig
