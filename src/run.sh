#!/usr/bin/env bash

JOB_DIR="/job-dir"
LOG_DIR="/job-dir"
DATASET_DIR="/dataset"
WORK_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CONFIG_DIR="${WORK_DIR}/configs"
ENCODER_DIR="datasets.writers"

if [ $1 = "train" ]; then
    python trainer.py \
        --config-path ${CONFIG_DIR}/default.yaml \
        "${@:2}"
elif [ $1 = "encode" ]; then
    python -m ${ENCODER_DIR}.$2 \
        --config-path ${CONFIG_DIR}/$2.yaml \
        "${@:3}"
elif [ $1 = "tensorboard" ]; then
    tensorboard --logdir=${LOG_DIR} --host=0.0.0.0 "${@:2}"
elif [ $1 = "notebook" ]; then
    jupyter notebook --generate-config
    echo "from notebook.auth import passwd; c.NotebookApp.password = passwd('${PASSWORD}')" >> ~/.jupyter/jupyter_notebook_config.py
    bash -c "source /etc/bash.bashrc && jupyter notebook --notebook-dir=/tf --ip 0.0.0.0 --no-browser --allow-root" "${@:2}"

elif [ $1 = "run" ]; then
    python "${@:2}"
else
    echo "Usage: run.sh [train|tensorboard|notebook|encode]"
    exit 1
fi