import os
import json
import base64
import requests
import tempfile
from typing import Dict, List, Any
import runpod
from PIL import Image
import uuid

# Import the original worker's generate function
from worker_runpod import generate

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
        # Get dimensions
        width, height = resolution_to_dimensions(job.get('res', '720p'))
        
        # Prepare input data for the original worker
        input_data = {
            "input": {
                "input_image": job['image_url'],
                "positive_prompt": job.get('prompt', 'beautiful scene'),
                "negative_prompt": job.get('negative_prompt', 'static, blurry, low quality'),
                "crop": "center",
                "width": width,
                "height": height,
                "length": job.get('frames', 53),
                "batch_size": 1,
                "shift": 8.0,
                "cfg": job.get('cfg', 1.0),
                "sampler_name": "lcm",
                "scheduler": "beta",
                "steps": job.get('steps', 4),
                "seed": job.get('seed', 0),
                "fps": job.get('fps', 24),
                "job_id": job_id
            }
        }
        
        # Call the original generate function directly
        result = generate(input_data)
        
        if result.get('status') == 'DONE':
            return {
                "status": "success",
                "job_id": job_id,
                "video_base64": result.get('base64_content', ''),
                "video_path": result.get('result', ''),
                "metadata": {
                    "resolution": f"{width}x{height}",
                    "fps": job.get('fps', 24),
                    "frames": job.get('frames', 53),
                    "duration": job.get('frames', 53) / job.get('fps', 24)
                }
            }
        else:
            return {
                "status": "error",
                "error": f"Generation failed: {result.get('result', 'Unknown error')}",
                "job_id": job_id
            }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "job_id": job_id
        }

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
        
        # Generate videos sequentially
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
        
        # Send webhook if provided
        if webhook_url:
            send_webhook(webhook_url, response)
        
        return response
        
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})