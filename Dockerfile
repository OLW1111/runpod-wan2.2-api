FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN git clone https://github.com/camenduru/wan2.2-i2v-rapid-tost.git .

RUN pip3 install --no-cache-dir \
    runpod \
    requests \
    Pillow \
    opencv-python \
    accelerate \
    transformers \
    diffusers \
    xformers

COPY worker_batch.py /workspace/worker_batch.py

RUN mkdir -p /workspace/output

CMD ["python3", "worker_batch.py"]