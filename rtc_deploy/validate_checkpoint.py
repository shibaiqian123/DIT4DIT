"""Saved-observation comparison. No server, camera or robot connection."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from model_adapter import Policy
from readonly_client import read_observation
from protocol import shifted


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('run','checkpoint','cosmos','capture','output'):p.add_argument('--'+k,required=True)
    a=p.parse_args();dest=Path(a.output);dest.mkdir(parents=True,exist_ok=False)
    policy=Policy(a.run,a.checkpoint,a.cosmos)
    obs=read_observation(Path(a.capture));ex=policy.example(obs)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        b=policy.model.backbone_interface
        features=b(**b.build_cosmos_inputs(images=[ex['image']],instructions=[ex['lang']]),
                   output_attentions=False,output_hidden_states=True,return_dict=True).hidden_states[-1].detach()
    state=torch.from_numpy(np.array([ex['state']])).to(features)
    torch.manual_seed(123)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float32):
        original=policy.model.action_model.predict_action(features,state)
    torch.manual_seed(123)
    with torch.autocast('cuda',dtype=torch.float32):
        zero=policy.rtc.predict_action_rtc(features,state,np.zeros((1,50,66)),10,25,guidance_alpha=0)
    # Fixed before comparison: exact deterministic parity for unchanged velocity path.
    torch.testing.assert_close(original,zero,rtol=0,atol=0)
    prefix=shifted(original.float().cpu().numpy()[0],25,10)
    records=[];allocated=[]
    for i in range(4):
        torch.manual_seed(100+i)
        plain=policy.predict(obs)
        torch.manual_seed(100+i)
        rtc=policy.predict(obs,prefix,10,25)
        n=np.asarray(rtc['normalized']);u=np.asarray(plain['normalized'])
        record=dict(normalized_overlap_rmse_plain=float(np.sqrt(np.mean((u[:10]-prefix[:10])**2))),
                    normalized_overlap_rmse_rtc=float(np.sqrt(np.mean((n[:10]-prefix[:10])**2))),
                    plain_seconds=plain['inference_seconds'],rtc_seconds=rtc['inference_seconds'],
                    missed_remaining_25frame_budget=rtc['inference_seconds']>.5,
                    rtc_out_of_range_fraction=rtc['out_of_range_fraction'])
        records.append(record)
        np.savez(dest/f'comparison_{i}.npz',previous=prefix,plain=u,rtc=n,
                 plain_tokens=plain['motion_tokens'],rtc_tokens=rtc['motion_tokens'])
        torch.cuda.synchronize();allocated.append(torch.cuda.memory_allocated())
    grads=any(p.grad is not None for p in policy.model.parameters())
    if grads:raise AssertionError('Model parameter gradients accumulated')
    report=dict(resources=policy.info,zero_guidance_max_abs=float((original-zero).abs().max()),
                parity_tolerance=dict(atol=0,rtol=0),records=records,
                cuda_allocated_bytes=allocated,parameter_gradients=grads,
                saved_observation=True,closed_loop=False,motor_send=False,
                notes='Repeated saved input measures sampler behavior, not physical continuity; timing excludes network.')
    (dest/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
