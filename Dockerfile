FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04
ARG IMAGE_VERSION=2025-09-18-01

# System deps
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Get your modified repo (assumes your fork has the needed nodes/code)
RUN git clone https://github.com/camenduru/wan2.2-i2v-rapid-tost.git .

# Python deps
# - Use headless OpenCV
# - Install ffmpeg-python (python module 'ffmpeg')
# - Install moviepy (you import VideoFileClip)
# - Install numpy explicitly
RUN pip3 install --no-cache-dir \
    runpod \
    requests \
    Pillow \
    numpy \
    opencv-python-headless \
    accelerate \
    transformers \
    diffusers \
    xformers \
    ffmpeg-python \
    moviepy

# (Optional) If your repo has requirements.txt for ComfyUI/nodes, do this too:
# RUN pip3 install --no-cache-dir -r requirements.txt || true

# Copy your changed files
COPY worker_batch.py /workspace/worker_batch.py
COPY worker_runpod.py /workspace/worker_runpod.py
COPY generate_video.py /workspace/generate_video.py

# Output dir used by your code
RUN mkdir -p /content/ComfyUI/output /content/ComfyUI/input

# (Optional but recommended) ensure models exist at expected paths.
# Adjust the URL / path to your weights location or mount them at runtime.
# Example placeholder:
# RUN mkdir -p /content/ComfyUI/models/checkpoints && \
#     echo "Put wan2.2-i2v-rapid-aio.safetensors here" > /content/ComfyUI/models/checkpoints/README

CMD ["python3", "worker_batch.py"]
RUN python3 - <<'PY'
import ffmpeg, cv2, moviepy, numpy
print("BUILD CHECK OK: ffmpeg, cv2, moviepy, numpy present")
PY
