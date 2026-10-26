#!/usr/bin/env bash
# Run from the DiT4DiT repository. No pretrained WBC action checkpoint is loaded.
set -euo pipefail
: "${COSMOS_MODEL:?Set COSMOS_MODEL to an existing local Cosmos-Predict2.5-2B directory}"
test -d "$COSMOS_MODEL"
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE="${WANDB_MODE:-offline}"
accelerate launch --config_file DiT4DiT/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${NUM_PROCESSES:-2}" DiT4DiT/training/train.py \
  --config_yaml DiT4DiT/config/real_robot/dit4dit_sonic_dex1.yaml \
  --framework.cosmos25.base_model "$COSMOS_MODEL" \
  --trainer.freeze_modules backbone_interface.extractor.text_encoder,backbone_interface.extractor.vae \
  --trainer.max_train_steps 16000 \
  --trainer.learning_rate.base 3e-5 \
  --trainer.learning_rate.vlm_interface 1e-5 \
  --trainer.learning_rate.action_model 1e-4 "$@"
