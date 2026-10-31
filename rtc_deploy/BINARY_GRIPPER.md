# Binary Dex1 inference output

`model_adapter.py` recognizes `action.dex1_label_encoding` in the run's
`dataset_statistics.json`. The run configuration must agree via
`datasets.vla_data.dex1_binary_gripper`. Missing/mismatched markers, unsupported
0/open and 1/closed semantics, or final action min/max other than 0/1 fail at
startup before checkpoint loading.

After the existing min-max denormalization, each of the last two scores is
thresholded at 0.5. Values below 0.5 request opening; values at or above 0.5
request closing. The server still sends motor-output radians in
`gripper_actions`, ordered left, right:

- Open targets: `[5.30725, 5.33774]` rad, the current calibrated full-open
  endpoints minus the 0.05 rad command margin.
- Closed targets: `[0.0, 0.0]` rad.

These endpoints are specific to the current Dex1 calibration. Update and validate
`DEX1_OPEN_TARGETS` if the robot calibration changes. Existing client/controller
limit checks, braking, obstruction handling and feedback checks remain active.

The 0.5 inference threshold differs from the training label hysteresis thresholds
0.2 and 0.8: those thresholds act on the recorded operator trigger. No sigmoid
is applied; these are denormalized regression scores, not calibrated probabilities.

Legacy runs without a binary marker and without the binary config flag keep
continuous motor-radian output. `normalized` and `motion_tokens` are unchanged,
including the normalized previous-action block used for RTC guidance. Both first
inference and RTC replanning share the output conversion. `/info` reports
`gripper_label_mode`, `gripper_output_units`, and the binary threshold/endpoints.

Offline tests (no checkpoint or robot connection):

```bash
cd rtc_deploy
python -m unittest test_gripper_output test_rtc test_service -v
```

Deploy the updated adapter from the training repository to the actual inference
server before using a binary-trained checkpoint. Updating a copy on host 229 does
not update or restart the server on host 32. Hardware behavior and inference with
a real binary checkpoint still require validation.
