# Use the official Miniconda3 base image (with CUDA support from NVIDIA)
FROM nvidia/cuda:11.8.0-base-ubuntu20.04

# Install system dependencies and Python
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    libglib2.0-0 \
    libx11-6 \
    libxext6 \
    libsm6 \
    libxrender1 \
    libssl-dev \
    curl \
    git \
    && apt-get clean

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda \
    && rm Miniconda3-latest-Linux-x86_64.sh

# Set Conda environment path
ENV PATH="/opt/conda/bin:$PATH"

# Create and activate Conda environment 'ensemble_classifier' with Python 3.8
RUN conda create -n ensemble_classifier python=3.8 -y

# Activate the environment, install PyTorch, torchvision, torchaudio, and pytorch-cuda
RUN /bin/bash -c "source activate ensemble_classifier && \
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y"

# Copy the current directory contents (including the requirements file) into the container
WORKDIR /app
COPY . /app

# Install additional Python dependencies using pip
RUN /bin/bash -c "source activate ensemble_classifier && \
    pip install -r requirements-classification.txt"

# Expose the port (if your application runs on a specific port)
EXPOSE 5000

# Command to run the application (replace app.py with your entry point)
CMD ["/bin/bash", "-c", "source activate ensemble_classifier && python app.py"]
