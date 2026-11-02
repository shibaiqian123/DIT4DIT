"""Two-rank CPU gather, duplicate handling, RNG/mode restoration and TB smoke."""
import json
import os
from pathlib import Path
import random
from types import SimpleNamespace
import numpy as np
import torch
import torch.distributed as dist
from test_sonic_kinematic_metrics import FakeDecoder,example,statistics
from DiT4DiT.evaluation.sonic_evaluator import SonicEvaluator
from DiT4DiT.training.trainer_utils.metric_logger import MetricLogger

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__();self.dropout=torch.nn.Dropout(.5)
    def predict_action(self,examples,**kwargs):
        assert not self.training and not self.dropout.training
        return {'normalized_actions':np.random.normal(0,.01,(1,50,66)).astype(np.float32)}


def run(e,model,writer=None):
    py=random.getstate();npstate=np.random.get_state();t=torch.get_rng_state().clone()
    values=e.run(model,123,writer)
    assert random.getstate()==py
    now=np.random.get_state();assert npstate[0]==now[0] and np.array_equal(npstate[1],now[1]) and npstate[2:]==now[2:]
    assert torch.equal(t,torch.get_rng_state())
    assert model.training and not model.dropout.training
    return values

if __name__=='__main__':
    dist.init_process_group('gloo');rank=dist.get_rank()
    e=SonicEvaluator.__new__(SonicEvaluator);e.config=SimpleNamespace(seed=42)
    e.dataset=[example(str(i)) for i in range(3)]
    e.dataset[-1]['valid_time_mask'][-7:]=False
    e.stats=statistics();e.decoder=FakeDecoder();e.metadata={'test':'synthetic_distributed'}
    e.output_dir=Path('results/sonic_kinematic_validation/distributed')
    model=Model().train();model.dropout.eval()
    cfg=SimpleNamespace(trackers=['tensorboard','jsonl'],output_dir=str(e.output_dir))
    logger=MetricLogger(cfg,rank==0)
    values=run(e,model,logger.writer);logger.log(values,123);logger.close()
    dist.barrier();dist.destroy_process_group()
    if rank==0:
        serial=run(e,model)
        assert values==serial,(values,serial)
        assert values['val/eval_windows']==3 and values['val/eef_kin_valid_frames']==143
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        events=EventAccumulator(str(e.output_dir/'tensorboard')).Reload()
        for k,v in values.items():
            event=events.Scalars(k)[-1];assert event.step==123 and np.isclose(event.value,v)
        result={'status':'DISTRIBUTED_SERIAL_RNG_MODE_AND_TENSORBOARD_OK','metrics':values}
        (e.output_dir/'verification.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result,indent=2))
