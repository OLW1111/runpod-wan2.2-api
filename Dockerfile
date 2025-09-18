# Use NVIDIA CUDA base image
FROM nvidia/cuda:11.8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    wget \
    curl \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Clone ComfyUI and WAN2.2 repository
RUN git clone https://github.com/camenduru/wan2.2-i2v-rapid-tost.git .

# Install Python dependencies
RUN pip3 install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 \
    runpod \
    requests \
    Pillow \
    opencv-python \
    accelerate \
    transformers \
    diffusers \
    xformers

# Copy custom worker
COPY worker_batch.py /workspace/worker_batch.py
COPY generate_video.py /workspace/generate_video.py

# Make output directory
RUN mkdir -p /workspace/output

# Set the command to run the worker
CMD ["python3", "worker_batch.py"]