# Start from CVT base image available at artifactory.
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# Install some basic utilities
RUN apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install -y \
    git \
    curl \
    wget \
    htop \
    iotop \
    tmux \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt /
RUN pip install -r /requirements.txt
RUN pip cache remove *

# Install .autoenv for automatic conda env activation and tab completion
RUN echo "autoenv() { [[ -f \"\$PWD/.autoenv\" ]] && source .autoenv ; } ; cd() { builtin cd \"\$@\" ; autoenv ; } ; autoenv" >> ~/.bashrc

# Add local files
COPY .project-root /workspace/
RUN mkdir /workspace/src
RUN mkdir /workspace/configs
RUN mkdir /workspace/scripts
RUN mkdir /workspace/data
RUN mkdir /workspace/logs

# Run jupyter lab in the background by default
EXPOSE 8888
CMD ["jupyter", "lab", "--allow-root", "--NotebookApp.token=''"]