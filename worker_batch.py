import os
import json
import base64
import requests
import subprocess
import tempfile
from typing import Dict, List, Any
import runpod
import torch
from PIL import Image
import uuid

# Environment variables
MODEL_PATH = "/workspace/models"
OUTPUT_PATH = "/workspace/output"

def download_image(url: str, filename: str) -> str:
    """Download image from URL and save locally"""
    response = requests.get(url)
    if response.status_code == 200:
        filepath = os.path.join("/tmp", filename)
        with open(filepath, "wb") as f:
            f.write(response.content)
        return filepath
    else:
        raise Exception(f"Failed to download image from {url}")

def resolution_to_dimensions(res: str) -> tuple:
    """Convert resolution string to width, height"""
    res_map = {
        "480p": (720, 480),
        "720p": (1280, 720),
        "1080p": (1920, 1080)
    }
    return res_map.get(res, (1280, 720))

def generate_single_video(job: Dict[str, Any], job_id: str) -> Dict[str, Any]:
    """Generate a single video from job parameters"""
    try:
        # Download image
        image_filename = f"{job_id}_{job['image']}.jpg"
        image_path = download_image(job['image_url'], image_filename)
        
        # Get dimensions
        width, height = resolution_to_dimensions(job.get('res', '720p'))
        
        # Prepare command
        cmd = [
            "python", "/workspace/generate_video.py",
            "--input-image", image_path,
            "--positive-prompt", job.get('prompt', 'beautiful scene'),
            "--negative-prompt", job.get('negative_prompt', 'static, blurry, low quality'),
            "--width", str(width),
            "--height", str(height),
            "--length", str(job.get('frames', 53)),
            "--fps", str(job.get('fps', 24)),
            "--steps", str(job.get('steps', 4)),
            "--cfg", str(job.get('cfg', 1.0)),
            "--seed", str(job.get('seed', 0))
        ]
        
        # Run generation
        result = subprocess.run(cmd, capture_output=True, text=True, cwd="/workspace")
        
        if result.returncode != 0:
            return {
                "status": "error",
                "error": f"Generation failed: {result.stderr}",
                "job_id": job_id
            }
        
        # Find generated video file
        output_files = [f for f in os.listdir("/workspace/output") if f.endswith('.mp4')]
        if not output_files:
            return {
                "status": "error", 
                "error": "No output video found",
                "job_id": job_id
            }
        
        video_path = os.path.join("/workspace/output", output_files[-1])
        
        # Convert to base64
        with open(video_path, "rb") as video_file:
            video_base64 = base64.b64encode(video_file.read()).decode('utf-8')
        
        return {
            "status": "success",
            "job_id": job_id,
            "video_base64": video_base64,
            "video_path": video_path,
            "metadata": {
                "resolution": f"{width}x{height}",
                "fps": job.get('fps', 24),
                "frames": job.get('frames', 53),
                "duration": job.get('frames', 53) / job.get('fps', 24)
            }
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "job_id": job_id
        }

def concatenate_videos(video_paths: List[str], output_path: str) -> str:
    """Concatenate multiple videos using ffmpeg"""
    try:
        # Create file list for ffmpeg
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            for path in video_paths:
                f.write(f"file '{path}'\n")
            filelist_path = f.name
        
        # Run ffmpeg concat
        cmd = [
            "ffmpeg", "-f", "concat", "-safe", "0",
            "-i", filelist_path,
            "-c", "copy",
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        os.unlink(filelist_path)
        
        if result.returncode != 0:
            raise Exception(f"FFmpeg concat failed: {result.stderr}")
            
        return output_path
        
    except Exception as e:
        raise Exception(f"Video concatenation failed: {str(e)}")

def send_webhook(webhook_url: str, data: Dict[str, Any]):
    """Send results to webhook URL"""
    try:
        requests.post(webhook_url, json=data, timeout=30)
    except Exception as e:
        print(f"Webhook failed: {str(e)}")

def handler(job):
    """Main handler function for RunPod"""
    try:
        job_input = job.get("input", {})
        jobs = job_input.get("jobs", [])
        concat_enabled = job_input.get("concat", False)
        webhook_url = job_input.get("webhook")
        
        if not jobs:
            return {"error": "No jobs provided"}
        
        # Generate videos in parallel (simulated - actual parallel processing would need threading)
        results = []
        video_paths = []
        
        for i, video_job in enumerate(jobs):
            job_id = f"job_{i}_{uuid.uuid4().hex[:8]}"
            result = generate_single_video(video_job, job_id)
            results.append(result)
            
            if result["status"] == "success":
                video_paths.append(result["video_path"])
        
        # Prepare response
        response = {
            "status": "completed",
            "results": results,
            "total_jobs": len(jobs),
            "successful_jobs": len([r for r in results if r["status"] == "success"])
        }
        
        # Handle concatenation if requested
        if concat_enabled and len(video_paths) > 1:
            try:
                concat_path = f"/workspace/output/concatenated_{uuid.uuid4().hex[:8]}.mp4"
                concatenate_videos(video_paths, concat_path)
                
                with open(concat_path, "rb") as f:
                    concat_base64 = base64.b64encode(f.read()).decode('utf-8')
                
                response["concatenated_video"] = {
                    "video_base64": concat_base64,
                    "path": concat_path
                }
            except Exception as e:
                response["concat_error"] = str(e)
        
        # Send webhook if provided
        if webhook_url:
            send_webhook(webhook_url, response)
        
        return response
        
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})