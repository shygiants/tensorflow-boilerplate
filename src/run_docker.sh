#!/usr/bin/env bash

PROJECT_NAME=TFBootstrap
CONTAINER_BASENAME=tf-bootstrap
DOCKERFILE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPOSITORY="$USER/$CONTAINER_BASENAME"
PORT=""
DEVICE_ID="7"
ARGS=""

function print-running() {
    if [ $1 = "tensorboard" ]; then
        echo "Running Tensorboard..."
    elif [ $1 = "encode" ]; then
        echo "Running encoding..."
    else
        echo "Running $PROJECT_NAME on GPU $2..."
    fi
}

# Check command
if { [ $1 = "train" ] || [ $1 = "export" ] || [ $1 = "run" ]; }; then
    if [ -z "$2" ]; then
        echo "Usage: run_docker.sh $1 DEVICE_ID"
        exit 1
    fi
    CONTAINER_NAME="$CONTAINER_BASENAME-$1-${2//,/-}"
    DEVICE_ID=$2
    ARGS="${@:3}"
elif [ $1 = "tensorboard" ]; then
    PORT="-p 6006:6006"
    CONTAINER_NAME="$CONTAINER_BASENAME-tensorboard"
elif [ $1 = "encode" ]; then
    CONTAINER_NAME="$CONTAINER_BASENAME-encode-$2"
    ARGS="${@:2}"
elif [ $1 = "build" ]; then
    echo "Only build Docker image."
else
    echo "Usage: run_docker.sh [train|export|run|tensorboard|encode|build] [DEVICE_ID]"
    exit 1
fi

# Build docker image
echo "Building Docker image..."
docker build -t ${REPOSITORY} ${DOCKERFILE_DIR}

if [ $1 = "build" ]; then
    exit 0
fi

print-running $@

# Check environment variables
if [ ! -f ${DOCKERFILE_DIR}/config.sh ]; then
    echo "config.sh not found! Make use of environment variables!"
    if [ -z "$JOB_DIR" ]; then
        UNSET_VARS=" JOB_DIR"
    fi
    if [ -z "$DATASET_DIR" ]; then
        UNSET_VARS="${UNSET_VARS} DATASET_DIR"
    fi
    if [ ! -z "$UNSET_VARS" ]; then
        echo "Environment variable$UNSET_VARS is not set"
        exit 1
    fi
else
    . ${DOCKERFILE_DIR}/config.sh
fi

# Remove current running container
docker stop ${CONTAINER_NAME}
docker rm ${CONTAINER_NAME}

# Run docker container
docker run --runtime=nvidia \
    -e "CUDA_VISIBLE_DEVICES=${DEVICE_ID}" \
    -e "JOB_DIR=/job-dir" \
    -e "LOG_DIR=/job-dir" \
    -e "DATASET_DIR=/dataset" \
    --name ${CONTAINER_NAME} \
    ${PORT} \
    -v ${JOB_DIR}:/job-dir -v ${DATASET_DIR}:/dataset \
    -d ${REPOSITORY} $1 ${ARGS}
