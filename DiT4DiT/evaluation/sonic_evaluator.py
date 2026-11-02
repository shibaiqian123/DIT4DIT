"""Deterministic offline-only validation with distributed sample de-duplication."""
import hashlib
import json
import random
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from .sonic_kinematics import SonicKinematicDecoder, sha256
from .sonic_metrics import KinematicMetrics


class SonicEvaluator:
    def __init__(self, config, dataset, output_dir, statistics_path=None):
        self.config, self.dataset = config, dataset
        self.output_dir=Path(output_dir)
        self.statistics_path=Path(statistics_path or self.output_dir/'dataset_statistics.json')
        stats=json.loads(self.statistics_path.read_text())
        self.stats=stats['new_embodiment']['action']
        self.decoder=SonicKinematicDecoder(config.checkpoint,config.sonic_root,
            eef_links=config.get('eef_links'), link_to_tcp=config.get('link_to_tcp'))
        self.metadata=dict(self.decoder.metadata, statistics_sha256=sha256(self.statistics_path),
            statistics_provenance=dataset.statistics_provenance,split=dataset.validation_manifest,
            metric_kind='decoded_label_kinematic_proxy',seed=int(config.seed),max_windows=len(dataset),
            prediction_settings={'sampler':'framework.predict_action; full Euler integration', 'rtc_history':False, 'postprocessing':'shared SONIC min_max inverse exactly once'}, token_mse_space='normalized_unclipped',
            kinematics_normalization='min_max_clipped',decoder_reference_indices=[0],
            decoder_reference_mask='current frame inherits valid action time; future references excluded')

    def run(self, model, step=0, writer=None):
        rank=dist.get_rank() if dist.is_initialized() else 0
        world=dist.get_world_size() if dist.is_initialized() else 1
        self.metadata["action_num_inference_timesteps"] = int(getattr(getattr(model,"action_model",None),"num_inference_timesteps",0))
        metrics=KinematicMetrics(self.decoder,self.stats)
        # Equal forward counts for sharded models. Duplicates use the same per-sample seed.
        count=len(self.dataset); padded=((count+world-1)//world)*world
        states=[(m,m.training) for m in model.modules()]
        py_state=random.getstate(); np_state=np.random.get_state()
        devices=list(range(torch.cuda.device_count()))
        error=None
        try:
            model.eval()
            with torch.random.fork_rng(devices=devices), torch.inference_mode():
                for offset in range(rank,padded,world):
                    index=offset%count
                    seed=int(self.config.seed)+index
                    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
                    ex=self.dataset[index]
                    output=model.predict_action(examples=[ex])
                    metrics.update(output['normalized_actions'],[ex])
        except Exception as exc:
            error=traceback.format_exc()
        finally:
            random.setstate(py_state); np.random.set_state(np_state)
            for module,training in states: module.training=training
        payload={'error':error,'rows':metrics.rows}
        gathered=[None]*world
        if world>1: dist.all_gather_object(gathered,payload)
        else: gathered=[payload]
        errors=[x['error'] for x in gathered if x['error']]
        if errors: raise RuntimeError('Kinematic evaluation failed: '+'; '.join(errors))
        combined={}
        for item in gathered:
            for key,row in item['rows'].items():
                if key in combined and any(not np.allclose(combined[key][k],row[k],atol=1e-5,rtol=1e-5) for k in row):
                    raise ValueError('Distributed duplicate prediction differs: '+key)
                combined[key]=row
        metrics.rows=combined
        values=metrics.compute()
        if rank==0:
            self.output_dir.mkdir(parents=True,exist_ok=True)
            (self.output_dir/'kinematic_evaluation_metadata.json').write_text(json.dumps(self.metadata,indent=2))
            from .sonic_trajectory import trajectory_figure
            import matplotlib.pyplot as plt
            # Fixed sample identity makes comparisons across training steps meaningful.
            key=next(k for k in sorted(metrics.rows) if metrics.rows[k]['valid'].any())
            figure=trajectory_figure(metrics.rows[key],key)
            try:
                figure.savefig(self.output_dir/f'eef_trajectory_step_{step}.png',dpi=140)
                if writer is not None:
                    writer.add_figure('val/eef_kin_trajectory',figure,step,close=False)
            finally:
                plt.close(figure)
            if writer is not None:
                writer.add_text('val/kinematic_reference',json.dumps(self.metadata),step)
                writer.flush()
        return values
