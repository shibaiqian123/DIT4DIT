# DiT4DiT test-time RTC deployment (read-only validation stage)

Git repository: `shibaiqian123/DIT4DIT`, branch `sonic`.
229 development checkout: `/home/cqu023/DiT4DiT/rtc_deploy`.
32 inference checkout: `/share/2025slx/DIT4DIT/rtc_deploy`.
This one-time recovery commits the existing32 source and pulls it onto229. Subsequent changes must be edited and committed on229, pushed, then pulled on32 with `git pull --ff-only origin sonic`. Environments, weights and validation artifacts stay local. The earlier GR00T deployment copy is retained for reference; do not maintain both copies independently.
This directory is independent of the old five-terminal launcher. No file here creates a DDS publisher, ZMQ robot command socket, or motor controller. `readonly_client.py` consumes actions into a JSONL log only. It does **not** implement SONIC execution or authorization to move a robot.

## Sampling contract

`RTCActionHead.predict_action_rtc` wraps the existing frozen DiT4DiT action head without changing its state_dict or trained architecture. It reuses its state encoder, action encoder, positional embedding, cross-attention model and decoder. Original model code / ordinary predict_action remain unchanged. The deployment framework adapter selects this independent sampler; it does not monkey-patch the old server.

Checkpoint resources must be explicit. Current contract is raw RGB image + original task text + raw `state[31]` -> normalized `50x66`. Use training state order and stats. The model's four Euler steps (noise time1 to0) remain unchanged. RTC guides **all66 normalized channels**. Training denormalization clips normalized values to[-1,1], applies min/max, and splits `motion_tokens[50,64]` and `gripper_actions[50,2]`. The clipping/out-of-range fraction is reported, not hidden. The 64 channels are checkpoint-specific latent actions, not joint angles.

Psi0's exponential mask and VJP are ported in rtc_sampler.py. For `x0=x-t*v`, `e=(previous-x0.detach())*mask`, compute `J(x0,x)^T e`; subtract the scaled correction from v, then take the original negative-time Euler step. Correction is zero at t=1 or near-zero VJP norm. Weight-one prefix is **soft guidance, not exact freezing**. No token interpolation or pseudo current-pose token is used. First chunk is ordinary inference.

Weights are frozen. Backbone and state features are detached. Only the per-step action sample carries gradients, and each step discards its graph. Do not wrap RTC in `torch.inference_mode()`. The original ordinary path is retained for baseline comparisons.

## Wire and timing

HTTP `/health`, `/info`; WebSocket `/ws`; server binds loopback8000. One active owner and one in-flight inference. An independent `observe` message replaces the single latest-observation slot. An `infer` request freezes its own observation and consumption position together; observations arriving mid-forward cannot change that request.

An observation has `image_png` (base64 PNG), `lang`, raw `state[31]`, `observation_id`, and `observed_at` (opaque client timestamp). An inference request adds `op=infer`, `session`, `request_id`, `chunk_id` (null initially), `consumed` (next unread index), `delay` (estimated consumed ticks during round trip). A reply includes the same request/session identity, new chunk_id, normalized actions, split denormalized outputs, source observation identity/time, resource hashes, inference duration, estimated delay and window_offset. No cross-host monotonic timestamp subtraction.

229's ConsumptionClock triggers after `action_exec_horizon` (default25) consumed rows. At request, `s` is actual consumed count; server shifts old normalized actions by s and pads the s-row tail with zero. That tail has **zero guidance**. Validate `0<=d<=s<=H-d` and s<H; no silent clamps. Delay window is six samples, initially10. On response, skip exactly the number of rows the client consumed while the request was pending; those rows belong to the committed old block. No burst catch-up. Client late by a full20ms tick fails instead of speeding up. Old-block exhaustion, expired response, invalid overlap or nonfinite output is terminal for that session. No infinite last-token hold.

Consumer progress is **not motor execution acknowledgement**. Future SONIC integration must preserve this distinction and its own stop/freshness checks. Saved-observation replay has neither live observation freshness nor closed-loop validity. RTC does not solve the initial physical takeover mismatch.

## Server32

Install only in an isolated environment layered on the existing model environment:

```bash
/share/2025slx/miniconda3/envs/dit4dit/bin/python -m venv --system-site-packages /share/2025slx/DIT4DIT/.venv_rtc_deploy
/share/2025slx/DIT4DIT/.venv_rtc_deploy/bin/python -m pip install --index-url https://pypi.org/simple -r /share/2025slx/DIT4DIT/rtc_deploy/requirements-server.txt
export DIT4DIT_ROOT=/share/2025slx/DIT4DIT
export RTC_PYTHON="$DIT4DIT_ROOT/.venv_rtc_deploy/bin/python"
# Choose a currently available A100 UUID from nvidia-smi, then set POLICY_GPU_UUID.
bash "$DIT4DIT_ROOT/rtc_deploy/start_server.sh" \
  --run "$DIT4DIT_ROOT/remote_policy/run19000" \
  --checkpoint steps_19000_pytorch_model.pt \
  --cosmos /share/2025slx/models/Cosmos-Predict2.5-2B
```

Port collision fails before model loading. No existing process is stopped. The service reports RTC_READY after resource loading; `/info` describes actual resources. Never infer a checkpoint from directory ordering. Source modifications to serve.py require restarting only this new service.

## Read-only client229

Tunnel in a separate terminal (don't create a duplicate):

```bash
ssh -N -T -o ExitOnForwardFailure=yes -L 127.0.0.1:8000:127.0.0.1:8000 2025slx@10.246.1.32
```

Use the isolated `.venv_rtc_client` environment with numpy,Pillow,websockets (no torch needed for the client). Install via `/home/cqu023/GR00T-WholeBodyControl/.venv_rtc_client/bin/python -m pip install -r /home/cqu023/DiT4DiT/rtc_deploy/requirements-client.txt`. Set `CAPTURE` to an existing saved capture containing manifest.json and observation.npz, and `RTC_OUTPUT` to a new directory:

```bash
cd /home/cqu023/DiT4DiT
/home/cqu023/GR00T-WholeBodyControl/.venv_rtc_client/bin/python rtc_deploy/readonly_client.py \
  --capture "$CAPTURE" --output "$RTC_OUTPUT" --blocks 3
```

Success marker: `READONLY_RTC_COMPLETE; NO MOTOR SEND`. Failure is recorded in events.jsonl; no automatic retry or threshold widening. The first response deadline is10seconds, matching the older offline request deadline; model loading happens before listening. Saved input is explicitly labeled, not presented as live control data.

## Tests and evidence

```bash
cd /share/2025slx/DIT4DIT/rtc_deploy
../.venv_rtc_deploy/bin/python -m unittest test_rtc test_service -v
```

`validate_checkpoint.py` accepts explicit --run, --checkpoint, --cosmos, --capture, --output. It fixes the zero-guidance parity tolerance at atol=rtol=0 before running, uses identical random noise, compares plain/RTC overlap RMSE and timing across four paired requests, records allocated CUDA memory and parameter gradients. Its repeated saved observation is an offline sampler comparison, **not** an executed robot trajectory. Allocated memory stability covers that short test only, not an indefinite leak proof.

`extract_validation_sample.py` can extract episode0/frame0 from the existing converted dataset using its recorded observation.sonic_state and first video frame. It records hashes, task, episode/frame and dataset-relative timestamp. It never substitutes zeros for missing data. Yesterday's missing manual capture is not reconstructed by this fixture.

See VALIDATION.md for measured results. Old workflows, checkpoints, data and SONIC protection settings are unchanged. Retirement of old entrypoints requires a later integration acceptance step.

## Psi0-style launch (added)

The existing `start_server.sh` now also accepts environment-only startup: explicitly set CHECKPOINT_DIR, CHECKPOINT_STEP and POLICY_GPU_UUID, then run `bash start_server.sh`. Its environment-only defaults are port8014, client execution horizon30, loopback host, and policydit4dit. The prior explicit-argument invocation remains compatible (serve.py defaults remain8000/25).

`serve.py --host 127.0.0.1 --port 8014 --action_exec_horizon 30 --policy dit4dit --rtc --run-dir "$CHECKPOINT_DIR" --ckpt-step "$CHECKPOINT_STEP" --cosmos "$COSMOS_PATH"` is supported. Step19000 resolves strictly to `checkpoints/steps_19000_pytorch_model.pt`; there is no latest/fallback checkpoint. `--rtc` explicitly selects the already-default test-time path, not train-time RTC.

Set the229 readonly client to `--url ws://127.0.0.1:8014/ws --action-exec-horizon 30` and tunnel8014 to8014. `/info` advertises the configured horizon and the server rejects premature replans. A window30 has20 frames remaining; previous validation used25, so it does not certify the30-frame real-network latency budget. Invalid delay/overlap still fails; no timeout is widened.

`--host 0.0.0.0` is supported if deliberately exposing the server directly. This unauthenticated diagnostic endpoint defaults to loopback + SSH forwarding.


## Deployment latency optimizations (2026-11-04)

- Startup polling yields outside the shared lock; heartbeat and gripper workers retain access.
- Frozen eval-mode prompt embeddings use a single-entry, per-model deployment cache. Set `RTC_DISABLE_PROMPT_CACHE=1` before server startup for an uncached baseline; `/info` reports `prompt_cache_enabled`. Model reload creates a new cache. Training code is unchanged.
- Exact repeated camera frames reuse PNG encoding only when image identity and source timestamp match. Body/gripper validation and original image age are retained. Already-224x224 server images skip redundant resizing.
- Live client `--diagnostics minimal` omits request image files, chunk dumps and per-frame token arrays, but retains frame indexes, ages, timing, stream changes and faults. Log flushes are batched up to 250 ms (fault/normal close flush); abrupt process termination may lose buffered lines. Default `full` retains replay inputs.
- No freshness, overlap, gain, limits or control deadline changes. Disable producer tracing explicitly with `unset LATENCY_TRACE_HOST LATENCY_TRACE_UDP LATENCY_TRACE_EVERY`; commented exports do not clear inherited values.

Validation on 229: syntax/help checks, no await under shared lock, CPU fake prompt-cache equality/invalidation, full/minimal recorder retention and unchanged infeasible-overlap rejection. No real-camera/DDS/motor execution. Real-checkpoint latency and output equivalence remain unmeasured.
For a read-only A/B, use the same saved observations/checkpoint/noise, keep full input records, separate warmup, compare request RTT and inference p50/p95/max plus overlap failures. Do not run an additional benchmark against a server controlling a robot. 32 receives changes only through Git; an already running server is not updated by pulling.

## Optional compute and wire optimizations (2026-11-04, second pass)

RTC sampling now computes the mask-presence predicate once per chunk and keeps
near-zero VJP normalization on device. Finite-value checks and the original
threshold remain in place. No horizon, timeout, freshness, joint or gripper
constraint is relaxed.

The following options are deliberately opt-in; existing launch commands and
full replies remain compatible:

- Server: `RTC_FEATURE_EARLY_EXIT=1` enables inference-only Cosmos feature-prefix
  execution. With extract_layer=17 this stops after block18 instead of executing
  all28 blocks and the output head. It requires a frozen eval extractor, capture
  at the first step, and no generated-video/future-loss request. The original
  training and video-generation paths remain unchanged. `/info` reports
  `feature_early_exit`. Keep this off until same-checkpoint, same-input/noise
  feature and final-action parity has been checked offline on the target GPU.
- Live/readonly sensor client: `--image-transport rgb` sends the same resized
  224x224 RGB uint8 pixels as base64 raw bytes instead of PNG. This avoids PNG
  compression/decompression but uses 200704 payload characters per observation;
  bandwidth can outweigh the CPU savings. Camera JPEG decoding is unchanged.
- Client: `--compact-replies` omits normalized50x66 from replies. The server
  validates and retains that block for subsequent RTC guidance; clients still
  validate the shape and finiteness of the transmitted token64 and gripper2.
  The server advertises both wire capabilities in `/info`; unsupported requested
  modes fail before creating execution publishers. Compact logs do not contain
  normalized blocks, so use full replies when those are needed for diagnosis.
- Existing `--diagnostics minimal` remains separate and optional. Keep full
  diagnostics for replay capture. No new option enables robot execution by itself.

Validation: existing10 CPU RTC tests on229 and4 in-memory service tests in a
temporary local environment; compact/full reply and reconnect tests for both
service copies;48 exact old/new guided-velocity comparisons
and NaN rejection; fake28-block prefix equivalence/hook cleanup/training guard;
PNG/raw pixel and state equality; compact reply negotiation, shape/finite checks
and stale-session rejection. Temporary validation files are not part of the
repository. These checks do not establish actual-checkpoint output equivalence,
GPU speedup or real-robot stability. Run a read-only A/B before enabling an
experimental path in execution; preserve identical inputs/noise and compare
p50/p95/max plus expired/infeasible-overlap counts. An active server must be
restarted by the operator after pulling to load new code; pulling alone does not
change the running process.

## Independent-GPU input / future-video viewer

`serve_preview.py` is an optional diagnostic process on32. The RTC server exposes
one latest inference-input snapshot only when started with `--enable-preview-source`.
The viewer polls this snapshot, uses a separate GPU/model/RNG, and never sends
robot actions. It has no video queue: intermediate inputs are skipped while a
video is generating. Both services must use a single full GPU UUID, and different
UUIDs. Checkpoint, config and statistics hashes must match. This consumes another
model's GPU memory; independently select an available GPU, not a training GPU.
Shared CPU/network overhead is still possible; this is not a zero-latency claim.

1. On32 keep the existing RTC launch options and add `--enable-preview-source`.
   Keep `CUDA_VISIBLE_DEVICES` set to the full UUID of the action GPU. Restart the
   action service only after stopping the robot/client through the existing flow.
2. In another32 terminal activate the same `.venv_rtc_deploy` environment and set
   `CHECKPOINT_DIR` and `CHECKPOINT_STEP` to exactly those used by the action
   service. Choose a different available GPU using `nvidia-smi -L`, then run:

   ```bash
   cd /share/2025slx/DIT4DIT
   source .venv_rtc_deploy/bin/activate
   export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
   export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
   # Set PREVIEW_GPU_UUID to the other available GPU's complete UUID first.
   python rtc_deploy/serve_preview.py \
     --run-dir "${CHECKPOINT_DIR:?Set the same run as RTC}" \
     --ckpt-step "${CHECKPOINT_STEP:?Set the same step as RTC}" \
     --cosmos /share/2025slx/models/Cosmos-Predict2.5-2B \
     --gpu-uuid "${PREVIEW_GPU_UUID:?Set a different available GPU UUID}" \
     --source-url http://127.0.0.1:8014 \
     --port 8015 --playback-fps 10
   ```

3. On229 keep the existing8014 action tunnel, and open an additional terminal:

   ```bash
   ssh -N -T -o ExitOnForwardFailure=yes \
     -L 127.0.0.1:8015:127.0.0.1:8015 2025slx@10.246.1.32
   ```

4. Open `http://127.0.0.1:8015/` in a229 browser. Start with the existing client in
   read-only mode (omit `--execute`). No new live-client arguments are needed.
   A frame appears only after an actual RTC request supplies a complete input.

The left panel holds the input frame corresponding to the video on the right;
it is not a separate live-camera feed. The right panel includes the conditional
frame and loops the generated clip. `--playback-fps` changes display speed only;
it does not infer physical timing. The page labels observation/request identity,
generation duration, input-snapshot age and result age. It shows unavailable/error
and stops playback when the source disconnects or goes stale. Late results from
an old RTC session are discarded. It does not replay past gripper/body commands.
Video generation is independent of RTC action sampling: the prediction is not a
rendering of the emitted50x66 action block or evidence of robot task success.

The explicit `predict_future=True` backbone path now correctly requests video
from the extractor. Default action feature extraction and the existing training
future-loss branch keep their prior video-return settings. Video generation uses
the checkpoint's configured future sampling steps and frame count; feature early
exit does not apply to this video path. No denoising-step increase is introduced.

Validation: syntax/help;4 original mock service tests;4 additional in-memory
checks covering GPU/resource guards, default-off source snapshots, pairing and
old-session result discard; fake-extractor branch checks on229. No new repository
test programs, real GPU generation, camera/DDS or motor validation were run.
Actual video quality, generation latency and shared-host impact remain unmeasured.
All implementation edits are on229;32 receives Git pulls only. This section does
not mean the code has already been pushed, pulled or loaded by a running service.

## Disable RTC guidance while retaining asynchronous delivery

Use `--no-rtc` instead of `--rtc` on the action server. The options are mutually
exclusive. Omitting both preserves the previous default (RTC enabled).

```bash
python rtc_deploy/serve.py \
  --host 127.0.0.1 --port 8014 \
  --action_exec_horizon 10 --policy dit4dit --no-rtc \
  --enable-preview-source \
  --run-dir "$CHECKPOINT_DIR" --ckpt-step "$CHECKPOINT_STEP" \
  --cosmos /share/2025slx/models/Cosmos-Predict2.5-2B
```

This command is for the32 inference checkout; the mirrored229 source entry is
`dit4dit/src/deploy/serve.py`. Keep the checkpoint, environment and full action-GPU
UUID unchanged. Stop the active robot/client through the existing flow before
restarting a server; the flag does not switch a running session in place.

`/info` must report `rtc_mode: "disabled"` and `rtc_enabled: false`.
`rtc_supported: true` is capability metadata, not an enabled-state indicator.
Every request now runs the original `model.predict_action`, including subsequent
chunks. There is no history-prefix guidance or VJP on this path, and the server
does not retain normalized history for RTC. Checkpoint loading, normalization,
64+2 action semantics and sampling step counts are unchanged.

The updated live and saved-observation clients read the mode from `/info` and
print `SAMPLING_MODE: disabled`; their launch commands do not need `--no-rtc`.
Update both client and server: old clients reject disabled mode. The ordinary
mode triggers replanning after `action_exec_horizon` consumed frames without
RTC-only d/s soft-mask overlap requirements. It still permits only one in-flight
request, skips the prefix elapsed during inference, rejects wrong-session/mode
replies, and stops on exhaustion or expired blocks. Existing live-observation
freshness, whole-tick deadlines, gripper limits and fault holding are unchanged.
Disabling RTC is not synchronous execution, nor a guarantee of chunk continuity.

The independent-GPU video service and its source endpoint remain compatible.
This mode changes action sampling only, not future-video generation.

Validation (no hardware/model checkpoint run):10 existing CPU RTC tests;
4 existing mock service tests;7 added temporary mode/clock/service checks on
each deployment copy; fake Policy calls proving every no-RTC request selects
ordinary sampling, plus live-client metadata validation. No repository test
programs were added. Actual checkpoint latency, boundary continuity and physical
behavior remain unmeasured. All edits are made on229;32 receives Git pulls only.
