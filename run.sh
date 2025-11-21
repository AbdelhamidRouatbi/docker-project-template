#!/usr/bin/env bash
set -e

# WANDB_API_KEY to be exported
docker run --rm \
  -p 5000:5000 \
  -e WANDB_API_KEY="${WANDB_API_KEY}" \
  ift6758-serving