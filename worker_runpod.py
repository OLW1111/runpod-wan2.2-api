import os
import json
import requests
import random
import time
import base64
from urllib.parse import urlsplit

import cv2               # needs opencv-python-headless
import ffmpeg            # python package "ffmpeg-python"
import torch
import numpy as np

# ComfyUI nodes
from nodes import NODE_CLASS_MAPPINGS
from comfy_extras import nodes_wan, nodes_model_advanced

# ---------------------------
# Config / model paths
# ---------------------------
CHECKPOINT_PATH = os.getenv(
    "WAN_CHECKPOINT_PATH",
    "wan2.2-i2v-rapid-aio.safetensors"  # override with absolute path if you place it under /content/ComfyUI/models/checkpoints/
)
CLIP_VISION_PATH = os.getenv(
    "WAN_CLIP_VISION_PATH",
    "clip_vision_vit_h.safetensors"
)

# Node instances
CheckpointLoaderSimple = NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]()
CLIPVisionLoader = NODE_CLASS_MAPPINGS["CLIPVisionLoader"]()
LoadImage = NODE_CLASS_MAPPINGS["LoadImage"]()
CLIPTextEncode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
CLIPVisionEncode = NODE_CLASS_MAPPINGS["CLIPVisionEncode"]()
WanImageToVideo = nodes_wan.NODE_CLASS_MAPPINGS["WanImageToVideo"]()
KSampler = NODE_CLASS_MAPPINGS["KSampler"]()
ModelSamplingSD3 = nodes_model_advanced.NODE_CLASS_MAPPINGS["ModelSamplingSD3"]()
VAEDecode = NODE_CLASS_MAPPINGS["VAEDecode"]()

# Globals for lazy load
_unet = _clip = _vae = _clip_vision = None

def _ensure_models_loaded():
    """Load heavy models once per process."""
    global _unet, _clip, _vae, _clip_vision
    if _unet is not None:
        return
    # Try a couple of common locations if relative path not found
    candidate_ckpts = [
        CHECKPOINT_PATH,
        f"/content/ComfyUI/models/checkpoints/{os.path.basename(CHECKPOINT_PATH)}"
    ]
    ckpt = next((p for p in candidate_ckpts if os.path.exists(p)), None)
    if ckpt is None:
        raise FileNotFoundError(
            f"WAN checkpoint not found. Tried: {candidate_ckpts}. "
            "Mount or download the weights and set WAN_CHECKPOINT_PATH."
        )

    candidate_clip = [
        CLIP_VISION_PATH,
        f"/content/ComfyUI/models/clip_vision/{os.path.basename(CLIP_VISION_PATH)}",
        f"/content/ComfyUI/models/clip/{os.path.basename(CLIP_VISION_PATH)}",
    ]
    clip_path = next((p for p in candidate_clip if os.path.exists(p)), None)
    if clip_path is None:
        raise FileNotFoundError(
            f"CLIP vision weights not found. Tried: {candidate_clip}. "
            "Mount or download and set WAN_CLIP_VISION_PATH."
        )

    with torch.inference_mode():
        _unet, _clip, _vae = CheckpointLoaderSimple.load_checkpoint(ckpt)
        _clip_vision = CLIPVisionLoader.load_clip(clip_path)[0]

def get_input_image_path(input_image: str) -> str:
    """Return local path for URL or local filename."""
    if not input_image.startswith(('http://', 'https://')):
        local_path = f"/content/ComfyUI/input/{input_image}"
        if os.path.exists(local_path):
            return local_path
        if os.path.exists(input_image):
            return input_image
        raise FileNotFoundError(f"Local image not found: {input_image}")

    os.makedirs("/content/ComfyUI/input", exist_ok=True)
    suffix = os.path.splitext(urlsplit(input_image).path)[1] or ".jpg"
    file_path = os.path.join("/content/ComfyUI/input", f"downloaded_image{suffix}")
    r = requests.get(input_image, timeout=60)
    r.raise_for_status()
    with open(file_path, "wb") as f:
        f.write(r.content)
    return file_path

def video_to_base64(video_path: str) -> str | None:
    try:
        with open(video_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print(f"[warn] base64 convert failed: {e}")
        return None

def images_to_mp4(images, output_path: str, fps: int = 24):
    """Write tensor images -> mp4 via ffmpeg-python."""
    frames = []
    for image in images:
        arr = 255.0 * image.cpu().numpy()
        img = np.clip(arr, 0, 255).astype(np.uint8)
        if img.shape[0] in [1, 3, 4]:
            img = np.transpose(img, (1, 2, 0))
        if img.shape[-1] == 4:
            img = img[:, :, :3]
        frames.append(img)

    temp_names = [f"temp_{i:04d}.png" for i in range(len(frames))]
    for name, frame in zip(temp_names, frames):
        if not cv2.imwrite(name, frame[:, :, ::-1]):  # BGR
            raise RuntimeError(f"Failed to write {name}")

    if not os.path.exists(temp_names[0]):
        raise RuntimeError("No temp PNGs were created")

    stream = ffmpeg.input("temp_%04d.png", framerate=fps)
    stream = ffmpeg.output(stream, output_path, vcodec="libx264", pix_fmt="yuv420p")
    ffmpeg.run(stream, overwrite_output=True)

    for name in temp_names:
        try:
            os.remove(name)
        except Exception:
            pass

@torch.inference_mode()
def generate(event_input: dict) -> dict:
    """
    RunPod-compatible handler (called by worker_batch).
    Expects: {"input": {...params...}}
    """
    _ensure_models_loaded()

    values = event_input["input"]

    input_image = values["input_image"]
    input_image_path = get_input_image_path(input_image)
    positive_prompt = values.get("positive_prompt", "")
    negative_prompt = values.get("negative_prompt", "")
    crop = values.get("crop", "center")
    width = int(values.get("width", 720))
    height = int(values.get("height", 480))
    length = int(values.get("length", 53))
    batch_size = int(values.get("batch_size", 1))
    shift = float(values.get("shift", 8.0))
    cfg = float(values.get("cfg", 1.0))
    sampler_name = values.get("sampler_name", "lcm")
    scheduler = values.get("scheduler", "beta")
    steps = int(values.get("steps", 4))
    seed = int(values.get("seed", 0))
    if seed == 0:
        random.seed(int(time.time()))
        seed = random.randint(1, 2**63 - 1)
    fps = int(values.get("fps", 24))

    # Compose graph
    model = ModelSamplingSD3.patch(_unet, shift)[0]
    positive = CLIPTextEncode.encode(_clip, positive_prompt)[0]
    negative = CLIPTextEncode.encode(_clip, negative_prompt)[0]

    start_img = LoadImage.load_image(input_image_path)[0]
    clip_vision_output = CLIPVisionEncode.encode(_clip_vision, start_img, crop)[0]
    positive, negative, out_latent = WanImageToVideo.encode(
        positive, negative, _vae, width, height, length, batch_size,
        start_image=start_img, clip_vision_output=clip_vision_output
    )
    out_samples = KSampler.sample(
        model, seed, steps, cfg, sampler_name, scheduler, positive, negative, out_latent
    )[0]
    decoded = VAEDecode.decode(_vae, out_samples)[0].detach()

    os.makedirs("/content/ComfyUI/output", exist_ok=True)
    out_path = f"/content/ComfyUI/output/wan2.2-i2v-rapid-{seed}-local.mp4"
    images_to_mp4(decoded, out_path, fps)

    job_id = values.get("job_id", f"local-job-{seed}")
    resp = {
        "jobId": job_id,
        "result": out_path,
        "status": "DONE",
        "message": "Video generated successfully",
    }
    b64 = video_to_base64(out_path)
    if b64:
        resp["base64_content"] = b64
        resp["base64_mime_type"] = "video/mp4"
    else:
        resp["message"] = "Video saved locally (base64 unavailable)"

    return resp

# IMPORTANT: do NOT start a server here. worker_batch will do it.
# if __name__ == "__main__":
#     import runpod
#     runpod.serverless.start({"handler": generate})
