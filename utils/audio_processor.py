import os
import tempfile
import yt_dlp
from pydub import AudioSegment

DOWNLOAD_DIR = 'downloades'
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def download_youtube_audio(url: str) -> str:
    output_path = os.path.join(DOWNLOAD_DIR, "%(id)s.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "noplaylist": True,
        "retries": 5,
        "fragment_retries": 5,
        "quiet": False,
        "cachedir": False,
        "verbose": True,
        "extractor_args": {
            "youtube": {
                "player_client": ["default,-android_sdkless"]
            }
        },
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
            }
        ],
    }

    # Optional YouTube cookie support
    cookie_file = None
    try:
        import streamlit as st
        if "YT_COOKIES" in st.secrets:
            cookie_content = st.secrets["YT_COOKIES"]
            if cookie_content and str(cookie_content).strip():
                with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
                    f.write(cookie_content)
                    cookie_file = f.name
    except Exception:
        pass

    if not cookie_file:
        local_cookie = "cookies.txt"
        if os.path.exists(local_cookie):
            cookie_file = local_cookie
        else:
            project_root_cookie = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cookies.txt"
            )
            if os.path.exists(project_root_cookie):
                cookie_file = project_root_cookie

    if cookie_file:
        ydl_opts["cookiefile"] = cookie_file

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if not info:
                raise RuntimeError("YouTube blocked this download. Use Upload File instead, or run the app locally.")
            raw_filename = ydl.prepare_filename(info)
            filename = os.path.splitext(raw_filename)[0] + ".wav"
    except Exception as e:
        err_msg = str(e).lower()
        if (
            "403" in err_msg
            or "forbidden" in err_msg
            or "index" in err_msg
            or "out of range" in err_msg
            or isinstance(e, IndexError)
        ):
            raise RuntimeError("YouTube blocked this download. Use Upload File instead, or run the app locally.") from e
        raise

    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"Audio download finished, but WAV file was not found: {filename}"
        )

    return filename


def convert_to_wav(input_path: str) -> str:
    """Convert any audio/video file to WAV format using pydub."""
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000)  # 16khz
    audio.export(output_path, format="wav")
    return output_path


def chunk_audio(wav_path: str, chunk_minutes: int = 10) -> list:
    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_minutes * 60 * 1000

    chunks = []
    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start: start + chunk_ms]
        chunk_path = f"{wav_path}_chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)

    if not chunks:
        # Fallback to single chunk if audio is very short or 0ms
        chunk_path = f"{wav_path}_chunk_0.wav"
        audio.export(chunk_path, format="wav")
        chunks.append(chunk_path)

    return chunks


def process_input(source: str) -> list:
    if source.startswith("http://") or source.startswith("https://"):
        print("Detected YouTube URL. Downloading audio...")
        wav_path = download_youtube_audio(source)
    else:
        print("Detected local file. Converting to WAV...")
        wav_path = convert_to_wav(source)

    print("Chunking audio...")
    chunks = chunk_audio(wav_path)
    print(f"Audio ready — {len(chunks)} chunk(s) created.")
    return chunks
