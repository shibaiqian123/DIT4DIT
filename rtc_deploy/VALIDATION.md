# Validation — 2026-10-30

## Scope and source

No robot commands were sent. This is sampler + cross-host read-only validation, not SONIC execution, startup continuity acceptance, or permission to resume hardware.

Checkpoint: `steps_19000_pytorch_model.pt`.
SHA256: `83271d85fc3b541b0437b960a0268cd51cb8e143e068e7f11a4f3d8de2bf7198`.
Config SHA256: `e6fe46f709bc801df9f0e413c90ebb196d68ae0d5bbb2245724a8db7c1e98a26`.
Stats SHA256: `b6a675604088c80298375e0b43fb9f2dfaa36203f8af4e94c34431f1d04e0463`.
Base path: `/share/2025slx/models/Cosmos-Predict2.5-2B` (base weight hashes are not included in this report).

Original manual capture directory from the previous conversation was absent at validation time. Instead, extracted converted training dataset `datasets/20261027/block_into_cup_v1_1_dex1_dit4dit`, episode0/frame0: recorded observation.sonic_state[31], first RGB video frame, task_index0. Fixture manifest includes parquet/video hashes. This is a real recorded dataset sample, NOT the prior failed startup and NOT live observations. Repeated observation tests cannot establish closed-loop behavior.

## Passed

- 14 unit / service tests: exact exponential mask, invalid overlap, analytic VJP, noise-time1, zero guidance/mask, no parameter grads, exact adapter Euler parity, no duplicate consumption, exhaustion/expiry failure, stale-session exclusion, reconnect, invalid block and concurrent request rejection.
- Actual checkpoint zero-guidance parity: max_abs_error=0, with predefined atol=rtol=0 and identical initial noise.
- Four paired ordinary/RTC requests with identical per-pair seeds: normalized prefix RMSE ordinary0.06661–0.07267, RTC0.03438–0.03929. This is latent-prefix agreement, not joint-space tracking or stability.
- Ordinary forward0.193–0.253s; RTC0.319–0.367s, excluding network. All four under remaining25/50=0.5s old-block budget.
- CUDA allocated memory identical across four measurements:21,439,843,840bytes. No accumulated model parameter gradients. This short run is not an indefinite leak proof.
- Raw normalized out-of-range fraction:2.24–2.88%. Existing training min/max denormalization clips to[-1,1]; clipping is retained and reported.

## Cross-host transport

After separating network send, response parsing and diagnostic file writes from the client consumption loop and disabling WebSocket compression, three mock blocks passed through229→SSH→32. Initial mock test had failed at a replan send due to client blocking; no time limit was widened to pass it.

Actual checkpoint service on temporary loopback18001, consumed by229:
- Three blocks; first ordinary prediction1.68356s (no action consumed while awaiting first block).
- RTC requests0.31705s and0.24612s; both launched with old-block position25.
- Initial delay estimate10, next estimate17; actual client-side skipped prefixes17 and13.
- 75 consumed (chunk_id,index) pairs,75 unique; maximum deadline lateness0.005467s.
- Final marker: `READONLY_RTC_COMPLETE; NO MOTOR SEND`.
- No backlog replay, no burst catch-up, no infinite final-token repetition.

The CLI intentionally stops after the requested number of blocks; it is not a long-run performance claim. The initial delay estimate underestimated the first observed round-trip delay (10 vs17); no hard-prefix equality is claimed. Additional live observation and SONIC integration tests remain necessary.

## Artifacts and cleanup

Preserved on229 under `outputs/dit4dit_rtc_validation_20261030/`: fixture, mock failure/pass receipts, actual3-block receipt, checkpoint comparison report and numeric arrays. On32 original numeric comparison remains in `rtc_deploy/validation_checkpoint_v1/`.

Temporary services PID1946492(mock18000) and1947937(real18001), and test SSH tunnels, were stopped after validation. Existing5560 inference and original deployment flows were not stopped or modified. The new server is not left running.

Final stricter protocol validation replayed all75 recorded consumer steps successfully, including reply-window and chunk identity checks.
