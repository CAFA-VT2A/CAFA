from ..data.utils import PadCrop

from torchaudio import transforms as T


from torch import nn
import math
import torch
import random
from typing import Tuple


def set_audio_channels(audio, target_channels):
    if target_channels == 1:
        # Convert to mono
        audio = audio.mean(1, keepdim=True)
    elif target_channels == 2:
        # Convert to stereo
        if audio.shape[1] == 1:
            audio = audio.repeat(1, 2, 1)
        elif audio.shape[1] > 2:
            audio = audio[:, :2, :]
    return audio


def prepare_audio(audio, in_sr, target_sr, target_length, target_channels, device):

    audio = audio.to(device)

    if in_sr != target_sr:
        resample_tf = T.Resample(in_sr, target_sr).to(device)
        audio = resample_tf(audio)

    audio = PadCrop(target_length, randomize=False)(audio)

    # Add batch dimension
    if audio.dim() == 1:
        audio = audio.unsqueeze(0).unsqueeze(0)
    elif audio.dim() == 2:
        audio = audio.unsqueeze(0)

    audio = set_audio_channels(audio, target_channels)

    return audio


def prepare_audio_gen(
    audio,
    in_sr,
    target_sr,
    target_length,
    target_channels,
    device,
    force_channels="stereo",
):
    audio = audio.to(device)
    pad_crop = PadCrop_Normalized_T(target_length, target_sr)

    encoding = torch.nn.Sequential(
        Stereo() if force_channels == "stereo" else torch.nn.Identity(),
        Mono() if force_channels == "mono" else torch.nn.Identity(),
    )
    audio, t_start, t_end, seconds_start, seconds_total, padding_mask = pad_crop(audio)

    # In training the auf is a pahse flipper. for us it is not relevant.
    # if self.augs is not None:
    #         audio = self.augs(audio)

    audio = audio.clamp(-1, 1)
    # Encode the file to assist in prediction

    if encoding is not None:
        audio = encoding(audio)

        ## in inference: adding batch dimention ###
    # Add batch dimension
    if audio.dim() == 1:
        audio = audio.unsqueeze(0).unsqueeze(0)
    elif audio.dim() == 2:
        audio = audio.unsqueeze(0)

    return audio


class PadCrop_Normalized_T(nn.Module):

    def __init__(self, n_samples: int, sample_rate: int, randomize: bool = True):

        super().__init__()

        self.n_samples = n_samples
        self.sample_rate = sample_rate
        self.randomize = randomize

    def __call__(
        self, source: torch.Tensor
    ) -> Tuple[torch.Tensor, float, float, int, int]:
        n_channels, n_samples = source.shape

        # If the audio is shorter than the desired length, pad it
        upper_bound = max(0, n_samples - self.n_samples)

        # If randomize is False, always start at the beginning of the audio
        offset = 0
        ## this is in case that the given audio is longer then the sample_size (which is 47 sec)
        ## The upper bound is for the offset element
        if self.randomize and n_samples > self.n_samples:
            offset = random.randint(0, upper_bound)

        # Calculate the start and end times of the chunk
        t_start = offset / (upper_bound + self.n_samples)
        t_end = (offset + self.n_samples) / (upper_bound + self.n_samples)

        # Create the chunk
        chunk = source.new_zeros([n_channels, self.n_samples])

        # Copy the audio into the chunk
        chunk[:, : min(n_samples, self.n_samples)] = source[
            :, offset : offset + self.n_samples
        ]

        # Calculate the start and end times of the chunk in seconds
        seconds_start = math.floor(offset / self.sample_rate)
        seconds_total = math.ceil(n_samples / self.sample_rate)

        # Create a mask the same length as the chunk with 1s where the audio is and 0s where it isn't
        padding_mask = torch.zeros([self.n_samples])
        padding_mask[: min(n_samples, self.n_samples)] = 1

        return (chunk, t_start, t_end, seconds_start, seconds_total, padding_mask)


class Mono(nn.Module):
    def __call__(self, signal):
        return (
            torch.mean(signal, dim=0, keepdims=True)
            if len(signal.shape) > 1
            else signal
        )


class Stereo(nn.Module):
    def __call__(self, signal):
        signal_shape = signal.shape
        # Check if it's mono
        if len(signal_shape) == 1:  # s -> 2, s
            signal = signal.unsqueeze(0).repeat(2, 1)
        elif len(signal_shape) == 2:
            if signal_shape[0] == 1:  # 1, s -> 2, s
                signal = signal.repeat(2, 1)
            elif signal_shape[0] > 2:  # ?, s -> 2,s
                signal = signal[:2, :]

        return signal
