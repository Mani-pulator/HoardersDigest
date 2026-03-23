import modal

app = modal.App("facebook-digest")

# Image with yt-dlp, whisper, ffmpeg. Git is needed for pip install from GitHub.
image = modal.Image.debian_slim().apt_install("git", "ffmpeg").run_commands(
    "pip install yt-dlp",
    "pip install git+https://github.com/openai/whisper.git",
    "pip install ffmpeg-python",
    "pip install torchvision",
    "pip install torch",
    "pip install accelerate",
    "pip install qwen-vl-utils",
    "pip install pillow",
    "pip install git+https://github.com/huggingface/transformers",
)



def download_video(url: str) -> str | None:
    
    import yt_dlp

    ydl_opts = {
        "format": "best",
        "outtmpl": "/tmp/%(id)s.%(ext)s",
        "noplaylist": True,  # don't download playlists
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            final_path = ydl.prepare_filename(info)
            ydl.download([url])
        return {
            "path": final_path,
            "description": info.get("description", ""),
            "title": info.get("title", "")
        }
    except Exception as e:
        print(f"Error downloading video: {e}")
        return None

def transcribe_video(video_path: str) -> str | None:

    import whisper
    try:
        model = whisper.load_model("base.en")
        transcription = model.transcribe(video_path)
        return transcription['text']
    except Exception as e:
        print(f"Whisper transcription failed: {e}")
        return None

def extract_frames(video_path: str) -> list[str]:
    import os
    import re
    import ffmpeg

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    frames_dir = f"/tmp/frames/{video_name}"
    os.makedirs(frames_dir, exist_ok=True)

    probe = ffmpeg.probe(video_path)
    video_stream = next(
        (s for s in probe["streams"] if s.get("codec_type") == "video"),
        probe["streams"][0],
    )
    width = int(video_stream["width"]) #can modify later to get smaller more efficient frames
    duration = float(
        probe.get("format", {}).get("duration")
        or video_stream.get("duration", 0)
    )

    parts = 5  # can modify later to get more or less frames
    if duration <= 0 or parts <= 0:
        return []

    interval_len = duration / parts
    frame_paths: list[str] = []
    for i in range(parts):
        t = min((i + 1) * interval_len, duration - 0.001)  # seek near end of each segment
        frame_path = f"{frames_dir}/frame_{i:04d}.jpg"
        (
            ffmpeg.input(video_path, ss=t)
            .filter("scale", width, -1)
            .output(frame_path, vframes=1)
            .run(overwrite_output=True, quiet=True)
        )
        if os.path.exists(frame_path):
            frame_paths.append(frame_path)
    return frame_paths

def describe_frames(video_path: str) -> str | None:
    # https://github.com/QwenLM/Qwen3-VL/blob/main/cookbooks/video_understanding.ipynb
    frame_paths = extract_frames(video_path)
    if not frame_paths:
        return None
    
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
    from PIL import Image

    model = Qwen3VLForConditionalGeneration.from_pretrained("Qwen/Qwen3-VL-2B-Instruct").to("cuda")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": frame_paths},
                {"type": "text", "text": "Describe this video."},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info(messages)

    inputs = processor(text=text, images=images, videos=videos, return_tensors="pt")
    inputs = inputs.to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=512)
    
    generated_ids_trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    output = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)
    return output[0]









@app.function(image=image, gpu="any")
async def process_video(url: str) -> str | None:
    video_info = download_video(url)
    if not video_info:
        return None
    transcription = transcribe_video(video_info["path"])
    visual_description = describe_frames(video_info["path"])
    return {
        "transcription": transcription,
        "visual_description": visual_description
    }

@app.local_entrypoint()
def main():
    print(process_video.remote("https://www.youtube.com/watch?v=dQw4w9WgXcQ"))