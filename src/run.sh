#!/usr/bin/env bash

JOB_DIR="/job-dir"
LOG_DIR="/job-dir"
DATASET_DIR="/dataset"
ENCODER_DIR="datasets.encoders"

if [ $1 = "train" ]; then
    python trainer.py \
        --job-dir $JOB_DIR \
        --dataset-dir $DATASET_DIR \
        "${@:2}"
elif [ $1 = "encode" ]; then
    python -m ${ENCODER_DIR}.$2 \
        --dataset-dir $DATASET_DIR \
        "${@:3}"
elif [ $1 = "tensorboard" ]; then
    tensorboard --logdir=${LOG_DIR} --host=0.0.0.0 "${@:2}"
elif [ $1 = "export" ]; then
    python exporter.py \
        --job-dir $JOB_DIR \
        "${@:2}"
elif [ $1 = "eval" ]; then
    python evaluator.py \
        --job-dir $JOB_DIR \
        --dataset-dir $DATASET_DIR \
        "${@:2}"
elif [ $1 = "notebook" ]; then
    /run_jupyter.sh --allow-root "${@:2}"
else
    echo "Usage: run.sh [train|export|eval|tensorboard|notebook|encode]"
    exit 1
fi