#!/bin/bash

# loads the project variables (namespace, etc.)
source project_vars.sh
# repo for images should be the same as the namespace
export REPO="${NAMESPACE}"
echo "Building docker image"
docker build --progress plain -f docker/Dockerfile -t ${REPO_HOST}/${REPO}:${PROJECT_NAME}-${DUSER}-v${VERSION} .
echo "Pushing docker container"
docker push ${REPO_HOST}/${REPO}:${PROJECT_NAME}-${DUSER}-v${VERSION}