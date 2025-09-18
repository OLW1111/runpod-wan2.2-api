FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04
ARG IMAGE_VERSION=2025-09-18-02
ENV PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y \
    git ffmpeg libgl1-mesa-glx libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# (Optional) Pull the reference repo you started from; harmless to keep
RUN git clone https://github.com/camenduru/wan2.2-i2v-rapid-tost.git . || true

# --- ComfyUI runtime (provides `nodes`, `comfy_extras`, etc.) ---
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /content/ComfyUI
RUN pip3 install --no-cache-dir -r /content/ComfyUI/requirements.txt || true

# Python deps (NO ffmpeg-python; we use system ffmpeg via subprocess)
RUN pip3 install --no-cache-dir \
    runpod requests Pillow numpy \
    opencv-python-headless \
    accelerate transformers diffusers xformers

# Make ComfyUI importable
ENV COMFY_PATH=/content/ComfyUI
ENV PYTHONPATH=/content/ComfyUI:${PYTHONPATH}

# Your modified files
COPY worker_batch.py   /workspace/worker_batch.py
COPY worker_runpod.py  /workspace/worker_runpod.py
COPY generate_video.py /workspace/generate_video.py

# IO dirs
RUN mkdir -p /content/ComfyUI/input /content/ComfyUI/output

# Build-time sanity check
RUN python3 - <<'PY'
import os, sys
sys.path.insert(0, "/content/ComfyUI")
import nodes
print("BUILD CHECK OK: ComfyUI nodes importable")
PY

# Single entrypoint — serverless starts here
CMD ["python3", "worker_batch.py"]
