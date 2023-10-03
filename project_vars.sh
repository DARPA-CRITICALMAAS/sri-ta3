#!/bin/bash

# the namespace created for your project
export NAMESPACE="criticalmaas-ta3"

################################################
# variables used in docker build/push script
################################################
# your project name
export PROJECT_NAME="cmaas-ta3"
export JOB_TAG= # meaninful name for THIS job - e.g. "-angel-dev"
export REPO_HOST="open.docker.sarnoff.com"
# docker containers will be tagged with your e##### and version
export VERSION=0.0
################################################
# variables used in run docker on k8s script
################################################
# requested machine configuration
export NGPU=1
export GPU_TYPE="2080Ti"
export TOTAL_CPU=8
export TOTAL_MEM=32
export MODE="run"
export SECRET="docker-io-secret"
################################################
# variables used in run docker locally script
################################################
export DATA_PATH= # [ABSOLUTE_PATH_TO_/k8s-data]
export LOGS_PATH= # [ABSOLUTE_PATH_TO_/k8s-logs]
export LOCAL_ROOT= # [ABSOLUTE_PATH_TO_/k8s-code]
export SRC_PATH=modeling/src
export CONFIG_PATH=modeling/configs
export SCRIPTS_PATH=modeling/scripts
