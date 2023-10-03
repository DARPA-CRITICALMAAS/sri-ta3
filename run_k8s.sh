#!/bin/bash

# loads the project variables (namespace, etc.)
source project_vars.sh

# repo for images should be the same as the namespace
export REPO="${NAMESPACE}"
# an already existing PVC (i.e., disk) in NAMESPACE or "" to use only emphemeral storage
export RW_VOLUME="${NAMESPACE}-rw"
export RO_VOLUME="${NAMESPACE}-ro"

envsubst < k8s/base/deployment.yaml > k8s/base/deployment.gen.yaml

kubectl apply -f k8s/base/deployment.gen.yaml