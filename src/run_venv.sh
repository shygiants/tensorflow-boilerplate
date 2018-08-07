#!/usr/bin/env bash

ENCODER_DIR="datasets.encoders"

if [ ! -f ../venv/bin/activate ]; then
    echo "Virtual environment not found!"
    echo "Prepare virtual environment."
    cd .. && bash ./prepare_venv.sh
fi

. ../venv/bin/activate

if [ $1 = "run" ]; then
    python $2 \
        --job-dir $JOB_DIR \
        --dataset-dir $DATASET_DIR \
        "${@:3}"
elif [ $1 = "train" ]; then
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
else
    echo "Usage: run.sh [run|train|encode|tensorboard|export]"
    exit 1
fi