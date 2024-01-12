#!/bin/bash
################################################
# docker project variables
################################################
export NAMESPACE="criticalmaas-ta3"
export PROJECT_NAME="cmaas-ta3"
export VERSION=0.0
################################################
# docker user variables
################################################
export JOB_TAG="-dev" # meaninful name for THIS job
export REPO_HOST="open.docker.sarnoff.com"
export DUSER=${USER}
export WANDB_API_KEY="00205bfc23fbecd8cd22f684188d4d9303f75d61"
################################################
# running docker locally variables
################################################
export PROJ_ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # default code path
export DATA_PATH=${PROJ_ROOT}/data # default data path
export LOGS_PATH=${PROJ_ROOT}/logs # default logs path
export SRC_PATH=sri-maper
################################################
# running docker on k8s variables
################################################
# requested machine configuration
export NGPU=1
export GPU_TYPE="2080Ti" # "2080Ti" "A5000"
export TOTAL_CPU=32
export TOTAL_MEM=32
export MODE="run"
export SECRET="docker-io-secret"
export K8_SRC_PATH=$(basename ${PROJ_ROOT})/${SRC_PATH}