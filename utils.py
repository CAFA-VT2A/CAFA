import random
import numpy as np
import torch
import subprocess
import tempfile
import logging
import os
from pathlib import Path
from shutil import which

# Configure logging for utils
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
if not log.hasHandlers():
    log.addHandler(logging.StreamHandler())


def seed_everything(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def apply_fn_recursive(obj, fn: callable):
    """Apply a function recursively to all tensors in a nested structure."""
    if isinstance(obj, torch.Tensor):
        return fn(obj)
    elif isinstance(obj, dict):
        return {k: apply_fn_recursive(v, fn) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [apply_fn_recursive(v, fn) for v in obj]
    elif isinstance(obj, tuple):
        return tuple([apply_fn_recursive(v, fn) for v in obj])
    else:
        raise NotImplementedError(f"obj type: {type(obj)}")


def load_video_ffmpeg(video_path, target_fps=25, target_size=256):
    """
    Load video using ffmpeg, resample to target_fps, resize, and extract frames.

    Args:
        video_path: Path to the video file.
        target_fps: Desired frame rate.
        target_size: Desired frame size (height and width).

    Returns:
        video_tensor: Tensor of shape [N, H, W, C] (normalized to [0, 1]).
        num_frames_read: Number of frames actually read from the video.
    """
    log.info(f"Processing video with ffmpeg: {video_path}")

    # Find ffmpeg path (using which_ffmpeg function from stable_audio_tools if available, otherwise rely on PATH)
    try:
        # Attempt to import and use which_ffmpeg if stable_audio_tools is importable
        from stable_audio_tools.utils.utils import which_ffmpeg

        ffmpeg_path = which_ffmpeg()
    except ImportError:
        log.warning(
            "stable_audio_tools.utils.utils.which_ffmpeg not found. Relying on ffmpeg being in PATH."
        )
        ffmpeg_path = "ffmpeg"  # Assume ffmpeg is in PATH

    # Use a temporary file for stderr to avoid pipe buffer issues with large outputs
    with tempfile.TemporaryFile(mode="w+") as stderr_file:
        command = [
            ffmpeg_path,
            "-i",
            str(video_path),
            "-r",
            str(target_fps),
            "-vf",
            f"scale={target_size}:{target_size}",
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "-",
        ]

        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=stderr_file)

        # Calculate expected size of one frame in bytes
        frame_size_bytes = target_size * target_size * 3  # H * W * C (RGB)

        # Read raw video data from stdout
        frames_data = process.stdout.read()
        process.wait()  # Wait for the process to finish

        # Read stderr output
        stderr_file.seek(0)
        stderr_output = stderr_file.read()

        if process.returncode != 0:
            log.error(f"ffmpeg error output:\n{stderr_output}")
            raise RuntimeError(f"ffmpeg failed with return code {process.returncode}")

        # Calculate number of frames actually read
        num_frames_read = len(frames_data) // frame_size_bytes
        log.info(f"Read {num_frames_read} frames from ffmpeg output.")

        if num_frames_read == 0:
            raise ValueError(
                "No frames could be extracted from the video using ffmpeg."
            )

        # Convert raw bytes to NumPy array
        # Handle potential partial frame data at the end
        valid_data_len = num_frames_read * frame_size_bytes
        video_array = np.frombuffer(frames_data[:valid_data_len], dtype=np.uint8)
        video_array = video_array.reshape(
            (num_frames_read, target_size, target_size, 3)
        )

        # Convert to float tensor and normalize to [0, 1]
        video_tensor = torch.from_numpy(video_array).float() / 255.0

    return video_tensor, num_frames_read


def combine_video_audio(video_path, audio_path, output_path, log):
    """Combines video and audio using ffmpeg."""
    try:
        # Check if ffmpeg is available
        if which("ffmpeg") is not None:
            mp4_output_path = Path(output_path).with_suffix(".mp4")

            # Use ffmpeg to combine original video with new audio
            cmd = [
                "ffmpeg",
                "-y",  # Overwrite output files without asking
                "-i",
                str(video_path),  # Input video file
                "-i",
                str(audio_path),  # Input audio file (wav)
                "-map",
                "0:v",  # Map video stream from first input
                "-map",
                "1:a",  # Map audio stream from second input
                "-c:v",
                "copy",  # Copy video stream without re-encoding
                "-c:a",
                "aac",  # Re-encode audio to AAC (common for MP4)
                "-shortest",  # Finish encoding when the shortest input stream ends
                str(mp4_output_path),
            ]

            log.info(
                f"Attempting to combine video and audio using command: {' '.join(cmd)}"
            )
            process = subprocess.run(cmd, check=True, capture_output=True, text=True)
            log.info(f"ffmpeg stdout:\n{process.stdout}")
            log.info(f"ffmpeg stderr:\n{process.stderr}")
            log.info(f"Video with generated audio saved to {mp4_output_path}")
        else:
            log.warning("ffmpeg not found, skipping video creation")
    except subprocess.CalledProcessError as e:
        log.error(f"Error running ffmpeg command: {e}")
        log.error(f"ffmpeg stdout:\n{e.stdout}")
        log.error(f"ffmpeg stderr:\n{e.stderr}")
    except Exception as e:
        log.error(f"Error creating video with new audio: {e}")
