import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchaudio
from omegaconf import OmegaConf

from stable_audio_tools.inference.generation import generate_diffusion_cond
from stable_audio_tools.models.pretrained import get_pretrained_model_local
from synchformer.utils.utils import which_ffmpeg

base_path = os.path.dirname(__file__)
synchformer_path = os.path.join(base_path, "synchformer")
sys.path.append(synchformer_path)

from utils import seed_everything, load_video_ffmpeg, combine_video_audio
from avclip_utils import (
    get_model as get_avclip_model,
    patch_config as patch_avclip_config,
    generate_embedding as generate_avclip_embedding,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)

TARGET_FPS = 25
MAX_GENERATION_DURATION_SEC = 10.0
FRAMES_FOR_10S = int(MAX_GENERATION_DURATION_SEC * TARGET_FPS)
TARGET_GEN_SEQ_LEN = 216

DEFAULT_CKPTS_DIR = Path("./ckpts")
DEFAULT_OUTPUT_DIR = Path("./output")
DEFAULT_EMBEDS_DIR = Path("./embeds")

MODEL_CONFIG_PATH = DEFAULT_CKPTS_DIR / "CAFA_avclip_config.json"
MODEL_PATH = DEFAULT_CKPTS_DIR / "CAFA_avclip.safetensors"
AVCLIP_CONFIG_PATH = DEFAULT_CKPTS_DIR / "avclip.yaml"
AVCLIP_MODEL_PATH = DEFAULT_CKPTS_DIR / "avclip.pt"


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate audio from video using Stable Audio with AV-CLIP embeddings"
    )
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--embed_path", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--embeds_dir", type=str, default="./embeds")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sigma_min", type=float, default=0.5)
    parser.add_argument("--sigma_max", type=float, default=500.0)
    parser.add_argument("--sampler_type", type=str, default="dpmpp-3m-sde")
    parser.add_argument("--asym_cfg", type=float, default=0.5)
    return parser.parse_args()


def setup_environment(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        log.warning("CUDA not available, running on CPU will be slow")

    seed_everything(args.seed)

    output_dir = Path(args.output_dir)
    embeds_dir = Path(args.embeds_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    embeds_dir.mkdir(parents=True, exist_ok=True)

    return device, output_dir, embeds_dir


def load_video_and_get_duration(video_path):
    try:
        video_tensor_full, num_frames_read = load_video_ffmpeg(
            video_path, target_fps=TARGET_FPS
        )
    except Exception as e:
        log.error(f"Failed to load video: {e}")
        sys.exit(1)

    actual_video_duration_sec = num_frames_read / TARGET_FPS
    target_audio_duration_sec = min(
        MAX_GENERATION_DURATION_SEC, actual_video_duration_sec
    )

    return video_tensor_full, target_audio_duration_sec


def prepare_video_for_embedding(video_tensor_full):
    current_frames = video_tensor_full.shape[0]
    video_tensor_for_embed = video_tensor_full

    if current_frames > FRAMES_FOR_10S:
        video_tensor_for_embed = video_tensor_full[:FRAMES_FOR_10S, ...]
    elif current_frames < FRAMES_FOR_10S:
        padding_size = FRAMES_FOR_10S - current_frames
        padding = torch.zeros(
            (padding_size, *video_tensor_for_embed.shape[1:]),
            dtype=video_tensor_for_embed.dtype,
            device=video_tensor_for_embed.device,
        )
        video_tensor_for_embed = torch.cat((video_tensor_for_embed, padding), dim=0)

    return video_tensor_for_embed


def load_avclip_model(device):
    try:
        avclip_config = OmegaConf.load(AVCLIP_CONFIG_PATH)
        avclip_config = patch_avclip_config(avclip_config)
        _, avclip_model = get_avclip_model(avclip_config, device)

        avclip_ckpt = torch.load(
            AVCLIP_MODEL_PATH, map_location="cpu", weights_only=False
        )
        state_dict = avclip_ckpt.get(
            "model", avclip_ckpt.get("state_dict", avclip_ckpt)
        )

        if all(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k[len("module.") :]: v for k, v in state_dict.items()}

        avclip_model.load_state_dict(state_dict)
        avclip_model = avclip_model.to(device)
        avclip_model.eval()

        return avclip_model, avclip_config
    except Exception as e:
        log.error(f"Failed to load AVClip model: {e}")
        return None, None


def get_avclip_embedding(args, video_tensor_full, device, embeds_dir):
    if args.embed_path and Path(args.embed_path).exists():
        try:
            av_clip_embedding = torch.from_numpy(np.load(args.embed_path)).to(device)
            return av_clip_embedding
        except Exception:
            log.warning(
                f"Failed to load embedding from {args.embed_path}, generating instead"
            )

    avclip_model, avclip_config = load_avclip_model(device)
    if avclip_model is None:
        log.error("Cannot generate embedding without AVClip model")
        return None

    video_tensor = prepare_video_for_embedding(video_tensor_full).to(device)

    try:
        av_clip_embedding = generate_avclip_embedding(
            avclip_model, avclip_config, video_tensor, device
        )

        video_path_obj = Path(args.video_path)
        embed_save_path = embeds_dir / f"{video_path_obj.stem}.npy"
        np.save(embed_save_path, av_clip_embedding.cpu().numpy())

        del avclip_model, avclip_config
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return av_clip_embedding
    except Exception as e:
        log.error(f"Error during embedding generation: {e}")
        return None


def prepare_embedding_for_generation(embedding, device):
    if embedding is None:
        return None

    embedding = embedding.to(device)
    if len(embedding.shape) == 2:
        embedding = embedding.unsqueeze(0)

    if len(embedding.shape) != 3:
        log.error(f"Unexpected embedding shape: {embedding.shape}")
        return None

    current_seq_len = embedding.shape[1]
    if current_seq_len < TARGET_GEN_SEQ_LEN:
        padding = torch.zeros(
            (1, TARGET_GEN_SEQ_LEN - current_seq_len, embedding.shape[2]),
            device=device,
            dtype=embedding.dtype,
        )
        embedding = torch.cat([embedding, padding], dim=1)
    elif current_seq_len > TARGET_GEN_SEQ_LEN:
        embedding = embedding[:, :TARGET_GEN_SEQ_LEN, :]

    return embedding


def load_generation_model(device):
    try:
        gen_model, gen_model_config = get_pretrained_model_local(
            str(MODEL_CONFIG_PATH), str(MODEL_PATH)
        )
        gen_model = gen_model.to(device)
        gen_model.eval()
        return gen_model, gen_model_config
    except Exception as e:
        log.error(f"Failed to load generation model: {e}")
        sys.exit(1)


def prepare_conditioning(args, embedding):
    conditioning = [
        {
            "prompt": args.prompt,
            "seconds_start": 0,
            "seconds_total": MAX_GENERATION_DURATION_SEC,
        }
    ]

    if embedding is not None:
        conditioning[0]["avclip_signal"] = embedding

    negative_conditioning = None
    if args.negative_prompt:
        negative_conditioning = [
            {
                "prompt": args.negative_prompt,
                "seconds_start": 0,
                "seconds_total": MAX_GENERATION_DURATION_SEC,
            }
        ]
        if embedding is not None:
            negative_conditioning[0]["avclip_signal"] = embedding

    return conditioning, negative_conditioning


def generate_audio_diffusion(
    model, config, conditioning, neg_conditioning, args, device
):
    sample_rate = config["sample_rate"]
    sample_size = config.get(
        "sample_size", int(sample_rate * MAX_GENERATION_DURATION_SEC)
    )

    try:
        output = generate_diffusion_cond(
            model,
            steps=args.steps,
            cfg_scale=args.cfg,
            conditioning=conditioning,
            negative_conditioning=neg_conditioning,
            sample_size=sample_size,
            sample_rate=sample_rate,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            sampler_type=args.sampler_type,
            asym_cfg=args.asym_cfg,
            seed=args.seed,
            batch_size=1,
            device=device,
        )
        return output, sample_rate
    except Exception as e:
        log.error(f"Error during audio generation: {e}")
        return None, sample_rate


def process_and_save_output(
    output_tensor, target_duration, sample_rate, output_dir, video_path
):
    if output_tensor is None:
        return

    max_val = torch.amax(torch.abs(output_tensor), dim=(1, 2), keepdim=True) + 1e-8
    output_norm = output_tensor / max_val
    output_norm = output_norm.clamp(-1, 1).mul(32767).to(torch.int16).cpu()

    max_length_samples = int(sample_rate * target_duration)
    if output_norm.shape[-1] > max_length_samples:
        output_norm = output_norm[..., :max_length_samples]

    video_path_obj = Path(video_path)
    output_name_stem = video_path_obj.stem
    wav_output_path = output_dir / f"{output_name_stem}.wav"

    try:
        torchaudio.save(
            str(wav_output_path), output_norm[0], sample_rate, channels_first=True
        )
        combine_video_audio(
            str(video_path_obj),
            str(wav_output_path),
            str(output_dir / f"{output_name_stem}_combined.mp4"),
            log,
        )
    except Exception as e:
        log.error(f"Failed to save output: {e}")


@torch.inference_mode()
def main():
    args = parse_arguments()
    device, output_dir, embeds_dir = setup_environment(args)

    # Process video and generate audio
    video_tensor, target_duration = load_video_and_get_duration(args.video_path)
    embedding = get_avclip_embedding(args, video_tensor, device, embeds_dir)

    del video_tensor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if embedding is None and not args.embed_path:
        log.error("Failed to generate or load embedding")
        return

    embedding = prepare_embedding_for_generation(embedding, device)
    gen_model, gen_config = load_generation_model(device)
    conditioning, neg_conditioning = prepare_conditioning(args, embedding)

    del embedding
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    audio, sample_rate = generate_audio_diffusion(
        gen_model, gen_config, conditioning, neg_conditioning, args, device
    )

    del gen_model, gen_config
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    process_and_save_output(
        audio, target_duration, sample_rate, output_dir, args.video_path
    )


if __name__ == "__main__":
    if which_ffmpeg() is None:
        log.error("ffmpeg not found in PATH")
        sys.exit(1)
    main()
