# loads base image
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# installs some basic utilities
RUN apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install -y \
    git \
    curl \
    wget \
    htop \
    iotop \
    tmux \
    vim \
    && rm -rf /var/lib/apt/lists/*

# installs GDAL
RUN conda install -c conda-forge "gdal>=3.0.2, <4.0.0"

# Install dependencies
RUN python3 -m pip install --upgrade pip wheel
COPY setup.py /workspace/
COPY README.md /workspace/
COPY sri-maper /workspace/sri-maper
RUN python3 -m pip install -e .

# Install .autoenv for automatic conda env activation and tab completion
# RUN echo "autoenv() { [[ -f \"\$PWD/.autoenv\" ]] && source .autoenv ; } ; cd() { builtin cd \"\$@\" ; autoenv ; } ; autoenv" >> ~/.bashrc

# Add local files
RUN mkdir /workspace/data
RUN mkdir /workspace/logs

# Run jupyter lab in the background by default
EXPOSE 8888
CMD ["jupyter", "lab", "--allow-root", "--NotebookApp.token=''"]