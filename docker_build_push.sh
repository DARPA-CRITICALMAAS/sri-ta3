#!/bin/bash

# loads the project variables (namespace, etc.)
source project_vars.sh
# repo for images should be the same as the namespace
export REPO="${NAMESPACE}"
echo "Building docker image"
docker build -f Dockerfile -t ${REPO_HOST}/${REPO}:${PROJECT_NAME}-${USER}-v${VERSION} .
echo "Pushing docker container"
docker push ${REPO_HOST}/${REPO}:${PROJECT_NAME}-${USER}-v${VERSION}