<div align="center">

# SRI DARPA AIE - CriticalMAAS TA3

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>

</div>

## Installation
This repo is compatible with running locally, on docker locally, or on docker in a Kubernetes cluster. Please **follow the corresponding instrcutions exactly, carefully** so install is smooth. Once you are familiar with the structure, you can make changes.

#### Local install and run
This setup presents the easiest installation but is more brittle than using docker containers. Please make a virtual environment of your choosing, source the environment, clone the repo, and install the code using `setup.py`. Below are example commands to do so.
```bash
# creates and activates virtual environment
conda create -n [VIRTUAL_ENV_NAME] python=3.10
conda activate [VIRTUAL_ENV_NAME]
# clone repo source code locally
git clone https://github.com/DARPA-CRITICALMAAS/sri-ta3.git
cd sri-ta3
# installs from source code
python3 -m pip install -e .
```
*If installation succeeded without errors,* you should be able to run the code locally. While we strongly encourage using the command-line-interface (CLI) to MAPER, we provide [example notebook files](./sri_maper/notebooks/) to demonstrate one might use the built CLI from a notebook or python file. You can view the example notebook files to get a sense of what the CLI is capable of. See [Command-Line-Interface Tutorial]() below for getting started with the CLI.

#### Install with docker container that is run locally
This setup is slightly more involved but provides more robustness across physical devices by using docker. We've written convenience [bash scripts](./docker) to make building and running the docker container much eaiser. First, edit the `JOB_TAG` `REPO_HOST`, `DUSER`, `WANDB_API_KEY` variables [project_vars.sh](project_vars.sh) to your use case. After editing [project_vars.sh](project_vars.sh), please clone the repo, and build the docker image. Below are example commands to do so using the conenivence scripts.
```bash
# clone repo source code locally
git clone https://github.com/DARPA-CRITICALMAAS/sri-ta3.git
cd sri-ta3
# builds docker image (installing source in image) and pushes to docker repo
bash docker/run_docker_build_push.sh
```
Optionally, if you would like to override the default `logs` and `data` folders within this repo that are empty to use exisitng ones (e.g. on datalake) that might contain existing logs and data, simply mount (or overwite) the corresponding folders on the datalake to the empty `logs` and `data` folders within this repo. Below are examles commands to do so.
```bash
sudo mount.cifs -o username=${USER},domain=sri,uid=$(id -u),gid=$(id -g) /datalake/path/to/existing/logs ./logs
sudo mount.cifs -o username=${USER},domain=sri,uid=$(id -u),gid=$(id -g) /datalake/path/to/existing/data ./data
```
*If installation succeeded without errors,* you should be able to run the code locally. While we strongly encourage using the command-line-interface (CLI) to MAPER, we provide [example notebook files](./sri_maper/notebooks/) to demonstrate one might use the built CLI from a notebook or python file. You can view the example notebook files to get a sense of what the CLI is capable of. Below are example commands to do so using the conenivence scripts. See [Command-Line-Interface Tutorial]() below for getting started with the CLI.
```bash
# starts the docker container
bash docker/run_docker_local.sh
##### EXECUTED WITHIN THE DOCKER CONTIAINER #####
# begins jupyter notebook
jupyter lab --ip 0.0.0.0 --allow-root --NotebookApp.token='' --no-browser
# now you can access the notebook files by browsing to http://localhost:8888/lab
```

#### Install with docker container that is run on the SRI International Kubernetes cluster
This setup is slightly more involved but provides more scalability to use more compute by using docker and Kubernetes. First we'll need to prepare some folders on the datalake to contain your data, code, and logs. Under the `criticalmaas-ta3` folder (namespace) within the `vt-open` datalake, make the following directory structure for YOUR use using your employee ID number (i.e. eXXXXX). NOTE, you only need to make the folders with the comment `CREATE` in it, the others should exist already. **Be careful not to corrupt the folders of other users or namespaces.**
```
vt-open
├── ... # other folders for other namespaces - avoid
├── criticalmaas-ta3 # top-level of criticalmaas-ta3 namespace
│   ├── data # contains all criticalmaas-ta3 data - (k8s READ ONLY)
│   └── k8s # contains criticalmaas-ta3 code & logs for ALL users - (k8s READ & WRITE)
│       ├── eXXXXX # folder you should CREATE to contain your code & logs
│       │   ├── code # folder you should CREATE to contain your code
│       │   └── logs # folder you should CREATE to contain your logs
│       └── ... # other folders for other users - avoid
└── ... # other folders for other namespaces - avoid
```
Next you will need to mount the `code` folder above locally. By mounting the `code` folder on the datalake locally, your local edits to source code will be reflected in the datalake, and therefore, on the Kubernetes cluster.
```bash
# makes a local code folder
mkdir k8s-code
# mount the datalake folder that hosts the code (Kubernetes will have access)
sudo mount.cifs -o username=${USER},domain=sri,uid=$(id -u),gid=$(id -g) /datalake/path/to/vt-open/criticalmaas-ta3/k8s/${USER}/code ./k8s-code
```
Last, we'll install the repo. We've written convenience [bash scripts](./docker) to make building and running the docker container much eaiser. Edit the `JOB_TAG` `REPO_HOST`, `DUSER`, `WANDB_API_KEY` variables [project_vars.sh](project_vars.sh) to your use case. After editing [project_vars.sh](project_vars.sh), please clone the repo, and build the docker image. Below are example commands to do so using the conenivence scripts.
```bash
# clone repo source code locally
git clone https://github.com/DARPA-CRITICALMAAS/sri-ta3.git
cd sri-ta3
# builds docker image (installing source in image) and pushes to docker repo
bash docker/run_docker_build_push.sh
```
*If installation succeeded without errors,* you should be able to run the code locally. While we strongly encourage using the command-line-interface (CLI) to MAPER, we provide [example notebook files](./sri_maper/notebooks/) to demonstrate one might use the built CLI from a notebook or python file. You can view the example notebook files to get a sense of what the CLI is capable of. Below are example commands to do so using the conenivence scripts. See [Command-Line-Interface Tutorial]() below for getting started with the CLI.
```bash
# starts the docker container
bash docker/run_docker_k8s.sh
# now you can access the notebook files by browsing to http://localhost:8888/lab
# note, you'll want to forward the Kubernetes container port 8888
```


## Command-Line-Interface (CLI) Tutorial

Using the CLI is the suggested method of integration into the MAPER code. As additional documentation, we provide [example notebook files](./sri_maper/notebooks/) that use the CLI internally within the jupyter notebook files. However, all actions performed in the jupyter notebook can be performed with the CLI (the notebooks just call the CLI functions internally). We suggest viewing the notebooks files as is (i.e. without running) to understand the CLI, then experiment with using the CLI directly. Below we give examples of the `train`, `test`, `map`, and `pretrain` capabilties through the CLI.

Train model with default configuration

```bash
# train on CPU
python sri_maper/src/train.py trainer=cpu

# train on GPU
python sri_maper/src/train.py trainer=gpu

# train on multi-GPU
python sri_maper/src/train.py trainer=ddp
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python sri_maper/src/train.py experiment=[example]
```

You can override any parameter from command line like this

```bash
python sri_maper/src/train.py trainer.max_epochs=20 data.batch_size=64
```

You can pretain a model like this

```bash
python sri_maper/src/pretrain.py ckpt_path=<PATH_TO_CHECKPOINT/*.ckpt>
```

You can test an existing checkpoint like this

```bash
python sri_maper/src/test.py ckpt_path=<PATH_TO_CHECKPOINT/*.ckpt>
```

You can output a map with prospectivity likelihood and uncertainty using an existing checkpoint like this (example in `exp_maniac_resnet_l22_uscont.yaml` experiment)

```bash
python sri_maper/src/map.py +experiment=exp_maniac_resnet_l22_uscont data.batch_size=128 ckpt_path=<PATH_TO_CHECKPOINT/*.ckpt>
```

#### To Do: DocStrings on entire repo; additional explanation of innerworkings of train, test, pretrain, map.