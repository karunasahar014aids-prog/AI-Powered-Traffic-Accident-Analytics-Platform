#!/usr/bin/env bash
set -e
mkdir -p data models
if [ "${TRAIN_MODEL:-false}" = "true" ]; then
  python train_model.py
else
  echo "Skipping full dataset training during deployment. Set TRAIN_MODEL=true to train the model."
fi
