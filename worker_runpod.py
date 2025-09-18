import os, sys, json, requests, random, time, base64, subprocess
from urllib.parse import urlsplit

import torch
import numpy as np
import cv2  # opencv-python-headless

# --- Make ComfyUI importable BEFORE using `nodes`
COMFY_PATH = os.getenv("COMFY_PATH", "/content/ComfyUI")
sys.path.insert(0, COMFY_PATH)
sys.path.insert(0, os.path.join(COMFY_PATH, "custom_nodes"))
if not os.path.exists(os.path.join(COMFY_PATH, "nodes.py")):
    raise FileNotFoundError(f"ComfyUI not found at {COMFY_PATH}. Clone it or set COMFY_PATH.")

from nodes import NODE_CLASS_MAPPINGS
from comfy_extras import nodes_wan, nodes_model_advanced

# Configurable model paths (env or default filenames searched in common dirs)
CHECKPOINT_PATH = os.getenv("WAN_CHECKPOINT_PATH", "wan2.2-i2v-rapid-aio.safetensors")
CLIP_VISION_PATH = os.getenv("WAN_CLIP_VISION_PATH", "clip_vision_vit_h.safetensors")

# Torch speed knobs (safe defaults)
torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

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

def _find_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

def _ensure_models_loaded():
    """Load heavy models once per process."""
    global _unet, _clip, _vae, _clip_vision
    if _unet is not None:
        return

    ckpt = _find_existing([
        CHECKPOINT_PATH,
        f"/content/ComfyUI/models/checkpoints/{os.path.basename(CHECKPOINT_PATH)}",
    ])
    if ckpt is None:
        raise FileNotFoundError(
            "WAN checkpoint not found. Put it at "
            "/content/ComfyUI/models/checkpoints/wan2.2-i2v-rapid-aio.safetensors "
            "or set WAN_CHECKPOINT_PATH to an absolute file path."
        )

    clip_path = _find_existing([
        CLIP_VISION_PATH,
        f"/content/ComfyUI/models/clip_vision/{os.path.basename(CLIP_VISION_PATH)}",
        f"/content/ComfyUI/models/clip/{os.path.basename(CLIP_VISION_PATH)}",
    ])
    if clip_path is None:
        raise FileNotFoundError(
            "CLIP vision weights not found. Put them at "
            "/content/ComfyUI/models/clip_vision/clip_vision_vit_h.safetensors "
            "or set WAN_CLIP_VISION_PATH."
        )

    with torch.inference_mode():
        _unet, _clip, _vae = CheckpointLoaderSimple.load_checkpoint(ckpt)
        _clip_vision = CLIPVisionLoader.load_clip(clip_path)[0]

def get_input_image_path(input_image: str) -> str:
    """Return local path for URL or local filename."""
    if not input_image.startswith(('http://', 'https://')):
        # local file
        local_path = f"/content/ComfyUI/input/{input_image}"
        if os.path.exists(local_path):
            return local_path
        if os.path.exists(input_image):
            return input_image
        raise FileNotFoundError(f"Local image not found: {input_image}")

    # download
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

def images_to_mp4(decoded_images, output_path: str, fps: int = 24):
    """
    Fast path: stream raw RGB frames to system ffmpeg (NVENC if available).
    Avoids writing hundreds of PNGs and avoids ffmpeg-python module.
    """
    frames = decoded_images.detach().cpu().numpy()  # [N,C,H,W] in 0..1 or 0..255
    if frames.shape[1] in (1, 3, 4):
        frames = np.transpose(frames, (0, 2, 3, 1))  # [N,H,W,C]
    if frames.shape[-1] == 4:
        frames = frames[..., :3]
    if frames.dtype != np.uint8:
        frames = np.clip(frames * 255.0, 0, 255).astype(np.uint8)

    n, h, w, c = frames.shape
    assert c == 3, "Expecting RGB frames"

    # Prefer NVENC; fall back to libx264 if NVENC not present
    codec = "h264_nvenc"
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-y",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}",
        "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", codec,
        "-preset", "p5",
        "-tune", "ll",
        "-rc", "vbr", "-b:v", "5M", "-maxrate", "10M",
        "-pix_fmt", "yuv420p",
        output_path,
    ]
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    except FileNotFoundError:
        # ffmpeg missing (shouldn't happen: installed via apt)
        raise RuntimeError("ffmpeg binary not found in container")
    try:
        proc.stdin.write(frames.tobytes(order="C"))
    finally:
        proc.stdin.close()
        proc.wait()
        if proc.returncode != 0:
            # Retry with CPU x264 once
            cmd_cpu = cmd[:]
            i_codec = cmd_cpu.index("-c:v")
            cmd_cpu[i_codec + 1] = "libx264"
            subprocess.run(cmd_cpu, input=frames.tobytes(order="C"), check=True)

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

# DO NOT start server here. `worker_batch.py` is the only entrypoint.
