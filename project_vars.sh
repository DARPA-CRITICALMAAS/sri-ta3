#!/bin/bash
################################################
# docker project variables
################################################
export NAMESPACE="criticalmaas-ta3"
export SYSTEM_NAME="sri-maper"
export SYSTEM_VERSION="0.0.1"
export SYSTEM_DESCRIPTION="SRI MAPER - Mineral Assessment Platform with Explainable Representations"
################################################
# the token that is used to authenticate with the CDR
################################################
export CDR_TOKEN="d25ad5e5c446573640612cc18b7536ba19e97e3a78de0edc22b15cf078360d25"
export CDR_HOST="https://api.cdr.land"
################################################
# docker user variables
################################################
export JOB_TAG="-dev-cdr-integration" # meaninful name for THIS job e.g. "-dev"
export REPO_HOST="open.docker.sarnoff.com" # docker repo e.g. "open.docker.sarnoff.com"
export DUSER=${USER} # username, default is for SRI's setup
export WANDB_API_KEY="b05b56b5af68d3c378d5ecfdf23ad3fbe997ad4a"
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
export TOTAL_MEM=64
export MODE="run"
export SECRET="docker-io-secret"
export K8_SRC_PATH=$(basename ${PROJ_ROOT})/${SRC_PATH}