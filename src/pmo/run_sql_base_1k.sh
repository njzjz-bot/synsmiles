#!/bin/bash
set -euo pipefail

cd /home/mila/m/minsu.kim/synsmiles/src/pmo

mkdir -p log

/home/mila/m/minsu.kim/envs/rxnflow/bin/python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device_name", torch.cuda.get_device_name(0))
PY

/home/mila/m/minsu.kim/envs/rxnflow/bin/python run.py sql_base \
  --oracles drd2 \
  --wandb disabled \
  --run_name sql_base_1k \
  --config_default hparams_sql_base.yaml \
  --seed 0 \
  --max_oracle_calls 1000 \
  --freq_log 100
