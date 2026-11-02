"""Frame-weighted wrist proxy metrics, independent of training or simulation."""
import numpy as np
import torch
from DiT4DiT.dataloader.gr00t_lerobot.sonic_normalization import unnormalize_sonic_actions


class KinematicMetrics:
    def __init__(self, decoder, stats):
        self.decoder, self.stats = decoder, stats
        self.rows = {}

    def update(self, prediction, examples):
        prediction = np.asarray(prediction, dtype=np.float32)
        labels = np.asarray([x['action'] for x in examples], dtype=np.float32)
        masks = np.asarray([x['action_mask'] for x in examples], dtype=bool)
        times = np.asarray([x['valid_time_mask'] for x in examples], dtype=bool)
        if prediction.shape != labels.shape or labels.shape[-1] != 66:
            raise ValueError('Expected matching [B,T,66] actions')
        if not np.isfinite(prediction).all() or not np.isfinite(labels).all():
            raise ValueError('Nonfinite action prediction/label')
        pred = unnormalize_sonic_actions(prediction, self.stats)[...,:64]
        target = unnormalize_sonic_actions(labels, self.stats)[...,:64]
        pp, pr = self.decoder.decode(pred)
        tp, tr = self.decoder.decode(target)
        pos = (pp-tp).norm(dim=-1)*100
        relative = pr.transpose(-1,-2)@tr
        # atan2 is accurate near both 0 and pi; avoid acos' identity-angle noise.
        vee = torch.stack((relative[...,2,1]-relative[...,1,2], relative[...,0,2]-relative[...,2,0], relative[...,1,0]-relative[...,0,1]),-1)
        sin = vee.norm(dim=-1)*.5
        cos = ((relative.diagonal(dim1=-2,dim2=-1).sum(-1)-1)*.5).clamp(-1,1)
        rot = torch.atan2(sin,cos).rad2deg()
        for i,ex in enumerate(examples):
            valid_elements = masks[i,:,:64] & times[i,:,None]
            valid_frames = masks[i,:,:64].all(-1) & times[i]
            key = ex['sample_id']
            row = dict(pos=pos[i].cpu().numpy(), rot=rot[i].cpu().numpy(), valid=valid_frames,
                squared=(prediction[i,:,:64]-labels[i,:,:64])**2,
                elements=valid_elements, outside=np.abs(prediction[i,:,:64])>1)
            if key in self.rows:
                old=self.rows[key]
                if any(not np.allclose(old[k],row[k]) for k in row):
                    raise ValueError('Conflicting predictions for duplicate sample '+key)
            self.rows[key]=row

    def compute(self):
        if not self.rows:
            raise ValueError('No validation samples')
        rows=[self.rows[key] for key in sorted(self.rows)]
        pos=np.stack([x['pos'] for x in rows]); rot=np.stack([x['rot'] for x in rows])
        valid=np.stack([x['valid'] for x in rows]); elements=np.stack([x['elements'] for x in rows])
        if not valid.any() or not elements.any():
            raise ValueError('No valid validation frames/token elements')
        sq=np.stack([x['squared'] for x in rows]); outside=np.stack([x['outside'] for x in rows])
        result={'val/token_mse':float(sq[elements].mean()), 'val/token_out_of_range_fraction':float(outside[elements].mean()),
                'val/eef_kin_valid_frames':int(valid.sum()),'val/eval_windows':len(rows)}
        for i,side in enumerate(('left','right')):
            base='val/eef_kin_'+side
            result[base+'_mean_cm']=float(pos[...,i][valid].mean())
            result[base+'_p95_cm']=float(np.quantile(pos[...,i][valid],.95))
            result[base+'_rot_deg']=float(rot[...,i][valid].mean())
            for start in range(0,pos.shape[1],10):
                stop=min(start+10,pos.shape[1]); mask=valid[:,start:stop]
                if mask.any():
                    result[f'{base}_mean_cm_h{start:02d}_{stop-1:02d}']=float(pos[:,start:stop,i][mask].mean())
        if not all(np.isfinite(v) for v in result.values()):
            raise ValueError('Nonfinite validation metric')
        return result
