#!/usr/bin/env python3
# ---------- BOOTSTRAP DEPS FIRST (before anything else) ----------
import sys, subprocess

def _ensure(pkg, import_name=None):
    import_name = import_name or pkg
    try:
        __import__(import_name)
        print(f"[deps] OK: {pkg} (import {import_name})")
    except Exception:
        print(f"[deps] MISSING: {pkg} -> installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", pkg])

# Required by worker_runpod.py at import-time
_ensure("ffmpeg-python", "ffmpeg")
_ensure("opencv-python-headless", "cv2")
_ensure("moviepy")
_ensure("numpy")
# -----------------------------------------------------------------

import os
import json
import uuid
from typing import Dict, Any, Tuple, List

import requests
import runpod

# Safe now: worker_runpod imports ffmpeg/cv2/etc during import
from worker_runpod import generate as generate_one


def resolution_to_dimensions(res: str) -> Tuple[int, int]:
    return {
        "480p": (720, 480),
        "720p": (1280, 720),
        "1080p": (1920, 1080),
    }.get(res, (1280, 720))


def generate_single_video(job: Dict[str, Any], job_id: str) -> Dict[str, Any]:
    try:
        width, height = resolution_to_dimensions(job.get("res", "720p"))
        input_payload = {
            "input": {
                "input_image": job["image_url"],
                "positive_prompt": job.get("prompt", "beautiful scene"),
                "negative_prompt": job.get("negative_prompt", "static, blurry, low quality"),
                "crop": "center",
                "width": width,
                "height": height,
                "length": int(job.get("frames", 53)),
                "batch_size": 1,
                "shift": 8.0,
                "cfg": float(job.get("cfg", 1.0)),
                "sampler_name": "lcm",
                "scheduler": "beta",
                "steps": int(job.get("steps", 4)),
                "seed": int(job.get("seed", 0)),
                "fps": int(job.get("fps", 24)),
                "job_id": job_id,
            }
        }
        result = generate_one(input_payload)

        if result.get("status") == "DONE":
            return {
                "status": "success",
                "job_id": job_id,
                "video_base64": result.get("base64_content", ""),
                "video_path": result.get("result", ""),
                "metadata": {
                    "resolution": f"{width}x{height}",
                    "fps": int(job.get("fps", 24)),
                    "frames": int(job.get("frames", 53)),
                    "duration": int(job.get("frames", 53)) / max(1, int(job.get("fps", 24))),
                },
            }
        else:
            return {
                "status": "error",
                "job_id": job_id,
                "error": f"Generation failed: {result.get('result', 'Unknown error')}",
            }
    except Exception as e:
        return {"status": "error", "job_id": job_id, "error": str(e)}


def send_webhook(webhook_url: str, data: Dict[str, Any]) -> None:
    try:
        requests.post(webhook_url, json=data, timeout=30)
    except Exception as e:
        print(f"[warn] webhook failed: {str(e)}")


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod serverless entry.
    Expects payload:
    {
      "input": {
        "jobs": [ {image_url, res, fps, frames, steps, cfg, prompt, seed}, ... ],
        "concat": false,
        "webhook": "https://example.com/hook"  # optional
      }
    }
    """
    try:
        job_input = job.get("input", {}) or {}
        jobs: List[Dict[str, Any]] = job_input.get("jobs", [])
        webhook_url = job_input.get("webhook") or job.get("webhook")

        if not jobs:
            return {"status": "error", "error": "No jobs provided"}

        results: List[Dict[str, Any]] = []
        video_paths: List[str] = []

        for idx, vjob in enumerate(jobs):
            jid = f"job_{idx}_{uuid.uuid4().hex[:8]}"
            res = generate_single_video(vjob, jid)
            results.append(res)
            if res.get("status") == "success" and res.get("video_path"):
                video_paths.append(res["video_path"])

        response = {
            "status": "completed",
            "results": results,
            "total_jobs": len(jobs),
            "successful_jobs": sum(1 for r in results if r.get("status") == "success"),
        }

        if webhook_url:
            send_webhook(webhook_url, response)

        return response

    except Exception as e:
        return {"status": "error", "error": str(e)}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
