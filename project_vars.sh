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
export JOB_TAG= # meaninful name for THIS job e.g. "-dev"
export REPO_HOST= # docker repo e.g. "open.docker.sarnoff.com"
export DUSER=${USER} # username, default is for SRI's setup
export WANDB_API_KEY="YOUR_WANDB_KEY"
################################################
# running docker locally variables
################################################
export PROJ_ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # default code path
export DATA_PATH=${PROJ_ROOT}/data # default data path
export LOGS_PATH=${PROJ_ROOT}/logs # default logs path
export SRC_PATH=sri_maper
################################################
# running docker on k8s variables
################################################
# requested machine configuration
export NGPU=1
export GPU_TYPE="A5000" # "2080Ti" "A5000"
export TOTAL_CPU=8
export TOTAL_MEM=40
export MODE="run"
export SECRET="docker-io-secret"
export K8_SRC_PATH=$(basename ${PROJ_ROOT})/${SRC_PATH}