#!/bin/bash

# loads the project variables (namespace, etc.)
source project_vars.sh

# repo for images should be the same as the namespace
export REPO="${NAMESPACE}"

# runs the container locally with access to GPU and proper volume mounts
docker run --rm -it --gpus all --name=cmaas-ta3-local -v ${DATA_PATH}:/workspace/data -v ${LOGS_PATH}:/workspace/logs -v ${LOCAL_ROOT}/${SRC_PATH}:/workspace/src -v ${LOCAL_ROOT}/${SCRIPTS_PATH}:/workspace/scripts -v ${LOCAL_ROOT}/${CONFIG_PATH}:/workspace/configs ${REPO_HOST}/${REPO}:${PROJECT_NAME}-${USER}-v${VERSION} bash