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
    "pip install bitsandbytes",
    "pip install git+https://github.com/huggingface/transformers",
)

summarize_image = modal.Image.debian_slim().pip_install("transformers", "torch", "accelerate")

INTERVAL_LEN = 5
FPS = 0.2


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

    print(f"Duration: {duration}")
    if duration <= 0:
        return []
    
    interval_len = INTERVAL_LEN
    
    parts = int(duration // interval_len)
    if parts <= 0:
        interval_len = interval_len//2
        parts = int(duration // interval_len)
    if parts <= 0:
        return []

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
                {"type": "video", "video": frame_paths, "fps": FPS},
                {"type": "text", "text": 
                "This is a short video the user saved because they found it valuable. What is the main topic or advice being communicated? What are the key points or takeaways? Only report what you observe directly. Do not interpret tone, intent, or make assumptions. Ignore visual aesthetics, focus only on the substance and meaning of the content. If there is not enough information, say 'not enough information'."},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

    inputs = processor(text=text, images=images, videos=videos, video_kwargs=video_kwargs, return_tensors="pt")
    inputs = inputs.to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=512)

    generated_ids_trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    output = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)
    return output[0]


def format_transcripts(saveTranscripts: list[dict]) -> str:
    output = []
    for i, r in enumerate(saveTranscripts):
        if r is None: 
            continue
        output.append(f"Video {i+1}: {r.get('title', 'No title')}")
        output.append(f"Description: {r.get('description', 'No description')}")
        output.append(f"Transcript: {r.get('transcription', 'No transcript')}")
        output.append(f"Visual: {r.get('visual_description', 'No visualdescription')}")
        output.append("")
    return "\n".join(output)



@app.function(image=summarize_image, gpu="any")
def summarize_collection(collection_name: str, saveTranscripts: list[dict]) -> str | None:
    import transformers
    import torch

    # model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    formatted_transcripts = format_transcripts(saveTranscripts)
    system_prompt = f"""You are an expert at synthesizing information. 
    You will be given transcripts from multiple short videos that a user saved in a collection called {collection_name}. 
    Your job is to extract and organize the key insights, advice, and information across all videos into a coherent summary.
    Based on the nature of this collection, choose the appropriate format and level of detail.
    Group related ideas together. Remove redundancy. Be thorough and detailed. Preserve important nuances, specific strategies, and actionable steps. Do not sacrifice detail for brevity.
    For advice or educational content, be thorough and preserve specific strategies and actionable steps.
    For recipes or tutorials, use structured step-by-step format.
    For general inspiration or entertainment, be concise.
    Do not sacrifice detail for brevity."""

    user_prompt = f"""Here are the transcripts from {len(saveTranscripts)} videos in the "{collection_name}" collection:
    {formatted_transcripts}
    Please provide an organized summary of the key insights across all these videos."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    

    terminators = [
        pipeline.tokenizer.eos_token_id,
        # pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        pipeline.tokenizer.convert_tokens_to_ids("<|/s|>")
    ]

    outputs = pipeline(
        messages,
        max_new_tokens=min(len(saveTranscripts) * 500, 8192),
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )

    print(outputs[0]["generated_text"][-1])

    




@app.function(image=image, gpu="any")
def process_video(url: str) -> str | None:
    video_info = download_video(url)
    if not video_info:
        return None
    transcription = transcribe_video(video_info["path"])
    visual_description = describe_frames(video_info["path"])
    return {
        "title": video_info["title"],
        "description": video_info["description"],
        "transcription": transcription,
        "visual_description": visual_description
    }

@app.function(image=image, gpu="any")
def process_collection(collection_id: int,collection_name: str, saves: list[dict]) -> str | None:
    processed_saves = list(process_video.map([s["url"] for s in saves]))
    summarized_collection = summarize_collection.remote(collection_name, processed_saves)
    pass

@app.local_entrypoint()
def main():
    #gotta hardcode for testing
    process_collection.remote(1, "career", 
        [
            {"url": "https://www.facebook.com/reel/1255321086186215/"},
            {"url": "https://www.facebook.com/reel/850301531156171/"},
            {"url": "https://www.facebook.com/reel/854445530630162/"},
            {"url": "https://www.facebook.com/reel/1467951958181218/"},
            {"url": "https://www.facebook.com/reel/1864924270875676/"},
        ]
    )

    
