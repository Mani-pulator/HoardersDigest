import modal

app = modal.App("facebook-digest")

# Image with yt-dlp, whisper, ffmpeg. Git is needed for pip install from GitHub.
image = modal.Image.debian_slim().apt_install("git", "ffmpeg").run_commands(
    "pip install yt-dlp",
    "pip install git+https://github.com/openai/whisper.git",
    "pip install ffmpeg-python",
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
        return final_path
    except Exception as e:
        print(f"Error downloading video: {e}")
        return None

def transcribe_video(video_path: str) -> str:
    # import ffmpeg
    # import re
    # audio_path = re.sub(r'\.[^.]+$', '.mp3', video_path)
    # try:
    #     ffmpeg.input(video_path).output(audio_path).run()
    #     print(f"Successfully converted {video_path} to {audio_path}")
    # except ffmpeg.Error as e:
    #     print(f"An error occurred: {e.stderr.decode()}")


    import whisper
    try:
        model = whisper.load_model("base.en")
        transcription = model.transcribe(video_path)
        return transcription['text']
    except Exception as e:
        print(f"Whisper transcription failed: {e}")
        return None


@app.function(image=image, gpu="any")
async def process_video(url: str) -> str | None:
    video_path = download_video(url)
    if not video_path:
        return None
    transcription = transcribe_video(video_path)
    return {
        "transcript": transcript,
    }


@app.local_entrypoint()
def main():
    print(process_video.remote("https://www.youtube.com/watch?v=dQw4w9WgXcQ"))