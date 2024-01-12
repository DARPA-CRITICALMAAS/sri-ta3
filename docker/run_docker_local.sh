#!/bin/bash

# loads the project variables (namespace, etc.)
source project_vars.sh

# repo for images should be the same as the namespace
export REPO="${NAMESPACE}"

# runs the container locally with access to GPU and proper volume mounts
docker run --rm -it --gpus all \
    --name=cmaas-ta3-local-${DUSER} \
    -v /dev/shm:/dev/shm \
    -v ${DATA_PATH}:/workspace/data \
    -v ${LOGS_PATH}:/workspace/logs \
    -v ${PROJ_ROOT}/${SRC_PATH}:/workspace/${SRC_PATH} \
    ${REPO_HOST}/${REPO}:${PROJECT_NAME}-${DUSER}-v${VERSION} bash