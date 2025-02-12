# Load base image
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# Install basic utilities
RUN apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install -y \
    git \
    curl \
    wget \
    htop \
    iotop \
    tmux \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Install GDAL
RUN conda install -c conda-forge "gdal>=3.0.2, <4.0.0"

# Install Python dependencies
RUN python3 -m pip install --upgrade pip wheel
COPY setup.py /workspace/
COPY README.md /workspace/
COPY sri_maper /workspace/sri_maper
RUN python3 -m pip install -e .

# Install .autoenv for automatic conda env activation and tab completion
RUN echo "autoenv() { [[ -f \"\$PWD/.autoenv\" ]] && source .autoenv ; } ; cd() { builtin cd \"\$@\" ; autoenv ; } ; autoenv" >> ~/.bashrc

# Add DOI Root Certificate
COPY docker/DOIRootCA2.crt /usr/local/share/ca-certificates
RUN chmod 644 /usr/local/share/ca-certificates/DOIRootCA2.crt && update-ca-certificates

# Set environment variables for HTTPS tools
ENV PIP_CERT="/etc/ssl/certs/ca-certificates.crt" \
    SSL_CERT_FILE="/etc/ssl/certs/ca-certificates.crt" \
    CURL_CA_BUNDLE="/etc/ssl/certs/ca-certificates.crt" \
    REQUESTS_CA_BUNDLE="/etc/ssl/certs/ca-certificates.crt" \
    AWS_CA_BUNDLE="/etc/ssl/certs/ca-certificates.crt"

# Create directories for local files
RUN mkdir /workspace/data
RUN mkdir /workspace/logs

# Copy the entrypoint script
COPY start.sh /workspace/start.sh
RUN chmod +x /workspace/start.sh

# Set the working directory
WORKDIR /workspace

# Expose port for JupyterLab
EXPOSE 8888

# Set the entrypoint to run the start script
ENTRYPOINT ["/workspace/start.sh"]
