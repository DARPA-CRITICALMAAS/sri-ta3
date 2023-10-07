<div align="center">

# SRI DARPA AIE - CriticalMAAS TA3

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>

</div>

## Installation
This repo is optimized for running on a Kubernetes (k8s) GPU cluster; however, it can be run through docker on the local workstation as well. Please **follow the instrcutions exactly, carefully** so install is smooth. Once you are familiar with the structure, you can make changes.
#### Prepare the directory structure on the datalake
First we'll need to prepare some folders on the datalake to contain your code and logs. Under the `criticalmaas-ta3` folder within the `vt-open` datalake, make the following directory structure for YOUR use using your employee ID number (i.e. eXXXXX). NOTE, you only need to make the folders with the comment `CREATE` in it, the others should exist already. **Be careful not to corrupt the folders of other users or namespaces.**
```
vt-open
├── ... # other folders for other namespaces - avoid
├── criticalmaas-ta3 # top-level of criticalmaas-ta3 namespace
│   ├── data # contains all criticalmaas-ta3 data - k8s can ONLY read
│   └── k8s # contains criticalmaas-ta3 code & logs for ALL users - k8s can read AND write
│       ├── eXXXXX # folder you should CREATE to contain your code & logs
│       │   ├── eXXXXX # folder you should CREATE to contain your code
│       │   └── eXXXXX # folder you should CREATE to contain your logs
│       └── ... # other folders for other users - avoid
└── ... # other folders for other namespaces - avoid
```
#### Prepare the repo with compatibility to run on k8s or locally
Run the follow in a directory on `vtsun-02` where you want have the project.
```bash
# first we'll make the folders to contain the code, logs, and data that k8s can access
mkdir k8s-code
mkdir k8s-logs
mkdir k8s-data
# next, we'll mount the datalake folders that hosts the code, logs, and data - which k8s will have access to
sudo mount.cifs -o username=${USER},domain=sri,uid=$(id -u),gid=$(id -g) //datalake-pr-smb.sri.com/vt-open/criticalmaas-ta3/k8s/${USER}/code ./k8s-code
sudo mount.cifs -o username=${USER},domain=sri,uid=$(id -u),gid=$(id -g) //datalake-pr-smb.sri.com/vt-open/criticalmaas-ta3/k8s/${USER}/logs ./k8s-logs
sudo mount.cifs -o username=${USER},domain=sri,uid=$(id -u),gid=$(id -g) //datalake-pr-smb.sri.com/vt-open/criticalmaas-ta3/data ./k8s-data
# next we'll grab the code from the gitlab repo and place into the newly generated code folder
cd ./k8s-code
git clone https://gitlab.sri.com/criticalmaas-ta3/modeling.git
cd ..
```
#### Build and run the docker container (on k8s or locally)
First, edit the `project_vars.sh` file to meet your needs. You can set it up to either use artifactory as a repository or a personal docker repo. With the `project_vars.sh` properly setup, the following can be run.
```bash
# builds the docker image and pushes to the repo
cd ./k8s-code
bash docker_build_push.sh
# starts a docker container on k8s
bash run_k8s.sh
# starts a docker container locally
bash run_local.sh
```
## How to run

Train model with default configuration

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu

# train on multi-GPU - AVOID (needs debug)
python src/train.py trainer=ddp
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=example
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 data.batch_size=64
```

You can test an existing checkpoint like this

```bash
python src/test.py ckpt_path=[PATH_TO_CHECKPOINT/*.ckpt]
```