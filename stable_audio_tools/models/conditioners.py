# Heavily influenced by https://github.com/facebookresearch/audiocraft/blob/main/audiocraft/modules/conditioners.py

import torch
import logging, warnings
import string
import typing as tp
import gc

from .adp import NumberEmbedder
from ..inference.utils import set_audio_channels
from .factory import create_pretransform_from_config
from .pretransforms import Pretransform
from ..utils.utils import copy_state_dict
from .utils import load_ckpt_state_dict

from torch import nn


class Conditioner(nn.Module):
    def __init__(self, dim: int, output_dim: int, project_out: bool = False):

        super().__init__()

        self.dim = dim
        self.output_dim = output_dim
        self.proj_out = (
            nn.Linear(dim, output_dim)
            if (dim != output_dim or project_out)
            else nn.Identity()
        )

    def forward(self, x: tp.Any) -> tp.Any:
        raise NotImplementedError()


class AVClipFrameConditioner(Conditioner):
    def __init__(
        self,
        output_dim: int,
    ):
        super().__init__(output_dim, output_dim)

    def forward(self, avclip_embed_control, device=None) -> tp.Any:
        return avclip_embed_control


class IntConditioner(Conditioner):
    def __init__(self, output_dim: int, min_val: int = 0, max_val: int = 512):
        super().__init__(output_dim, output_dim)

        self.min_val = min_val
        self.max_val = max_val
        self.int_embedder = nn.Embedding(
            max_val - min_val + 1, output_dim
        ).requires_grad_(True)

    def forward(self, ints: tp.List[int], device=None) -> tp.Any:

        # self.int_embedder.to(device)

        ints = torch.tensor(ints).to(device)
        ints = ints.clamp(self.min_val, self.max_val)

        int_embeds = self.int_embedder(ints).unsqueeze(1)

        return [int_embeds, torch.ones(int_embeds.shape[0], 1).to(device)]


class NumberConditioner(Conditioner):
    """
    Conditioner that takes a list of floats, normalizes them for a given range, and returns a list of embeddings
    """

    def __init__(self, output_dim: int, min_val: float = 0, max_val: float = 1):
        super().__init__(output_dim, output_dim)

        self.min_val = min_val
        self.max_val = max_val

        self.embedder = NumberEmbedder(features=output_dim)

    def forward(self, floats: tp.List[float], device=None) -> tp.Any:

        # Cast the inputs to floats
        floats = [float(x) for x in floats]

        floats = torch.tensor(floats).to(device)

        floats = floats.clamp(self.min_val, self.max_val)

        normalized_floats = (floats - self.min_val) / (self.max_val - self.min_val)

        # Cast floats to same type as embedder
        embedder_dtype = next(self.embedder.parameters()).dtype
        normalized_floats = normalized_floats.to(embedder_dtype)

        float_embeds = self.embedder(normalized_floats).unsqueeze(1)

        return [float_embeds, torch.ones(float_embeds.shape[0], 1).to(device)]


class T5Conditioner(Conditioner):

    T5_MODELS = [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
        "google/flan-t5-small",
        "google/flan-t5-base",
        "google/flan-t5-large",
        "google/flan-t5-xl",
        "google/flan-t5-xxl",
    ]

    T5_MODEL_DIMS = {
        "t5-small": 512,
        "t5-base": 768,
        "t5-large": 1024,
        "t5-3b": 1024,
        "t5-11b": 1024,
        "t5-xl": 2048,
        "t5-xxl": 4096,
        "google/flan-t5-small": 512,
        "google/flan-t5-base": 768,
        "google/flan-t5-large": 1024,
        "google/flan-t5-3b": 1024,
        "google/flan-t5-11b": 1024,
        "google/flan-t5-xl": 2048,
        "google/flan-t5-xxl": 4096,
    }

    def __init__(
        self,
        output_dim: int,
        t5_model_name: str = "t5-base",
        max_length: str = 128,
        enable_grad: bool = False,
        project_out: bool = False,
    ):
        assert (
            t5_model_name in self.T5_MODELS
        ), f"Unknown T5 model name: {t5_model_name}"
        super().__init__(
            self.T5_MODEL_DIMS[t5_model_name], output_dim, project_out=project_out
        )

        from transformers import T5EncoderModel, AutoTokenizer

        self.max_length = max_length
        self.enable_grad = enable_grad

        # Suppress logging from transformers
        previous_level = logging.root.manager.disable
        logging.disable(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                # self.tokenizer = T5Tokenizer.from_pretrained(t5_model_name, model_max_length = max_length)
                # model = T5EncoderModel.from_pretrained(t5_model_name, max_length=max_length).train(enable_grad).requires_grad_(enable_grad)
                self.tokenizer = AutoTokenizer.from_pretrained(t5_model_name)
                model = (
                    T5EncoderModel.from_pretrained(t5_model_name)
                    .train(enable_grad)
                    .requires_grad_(enable_grad)
                    .to(torch.float16)
                )
            finally:
                logging.disable(previous_level)

        if self.enable_grad:
            self.model = model
        else:
            self.__dict__["model"] = model

    def forward(
        self, texts: tp.List[str], device: tp.Union[torch.device, str]
    ) -> tp.Tuple[torch.Tensor, torch.Tensor]:

        self.model.to(device)
        self.proj_out.to(device)

        encoded = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device).to(torch.bool)

        self.model.eval()

        with torch.cuda.amp.autocast(dtype=torch.float16) and torch.set_grad_enabled(
            self.enable_grad
        ):
            embeddings = self.model(input_ids=input_ids, attention_mask=attention_mask)[
                "last_hidden_state"
            ]

        embeddings = self.proj_out(embeddings.float())

        embeddings = embeddings * attention_mask.unsqueeze(-1).float()

        return embeddings, attention_mask


class PhonemeConditioner(Conditioner):
    """
    A conditioner that turns text into phonemes and embeds them using a lookup table
    Only works for English text

    Args:
        output_dim: the dimension of the output embeddings
        max_length: the maximum number of phonemes to embed
        project_out: whether to add another linear projection to the output embeddings
    """

    def __init__(
        self,
        output_dim: int,
        max_length: int = 1024,
        project_out: bool = False,
    ):
        super().__init__(output_dim, output_dim, project_out=project_out)

        from g2p_en import G2p

        self.max_length = max_length

        self.g2p = G2p()

        # Reserving 0 for padding, 1 for ignored
        self.phoneme_embedder = nn.Embedding(len(self.g2p.phonemes) + 2, output_dim)

    def forward(
        self, texts: tp.List[str], device: tp.Union[torch.device, str]
    ) -> tp.Tuple[torch.Tensor, torch.Tensor]:

        self.phoneme_embedder.to(device)
        self.proj_out.to(device)

        batch_phonemes = [
            self.g2p(text) for text in texts
        ]  # shape [batch_size, length]

        phoneme_ignore = [" ", *string.punctuation]

        # Remove ignored phonemes and cut to max length
        batch_phonemes = [
            [p if p not in phoneme_ignore else "_" for p in phonemes]
            for phonemes in batch_phonemes
        ]

        # Convert to ids
        phoneme_ids = [
            [self.g2p.p2idx[p] + 2 if p in self.g2p.p2idx else 1 for p in phonemes]
            for phonemes in batch_phonemes
        ]

        # Pad to match longest and make a mask tensor for the padding
        longest = max([len(ids) for ids in phoneme_ids])
        phoneme_ids = [ids + [0] * (longest - len(ids)) for ids in phoneme_ids]

        phoneme_ids = torch.tensor(phoneme_ids).to(device)

        # Convert to embeddings
        phoneme_embeds = self.phoneme_embedder(phoneme_ids)

        phoneme_embeds = self.proj_out(phoneme_embeds)

        return phoneme_embeds, torch.ones(
            phoneme_embeds.shape[0], phoneme_embeds.shape[1]
        ).to(device)


class TokenizerLUTConditioner(Conditioner):
    """
    A conditioner that embeds text using a lookup table on a pretrained tokenizer's vocabulary

    Args:
        tokenizer_name: the name of the tokenizer from the Hugging Face transformers library
        output_dim: the dimension of the output embeddings
        max_length: the maximum length of the text to embed
        project_out: whether to add another linear projection to the output embeddings
    """

    def __init__(
        self,
        tokenizer_name: str,  # Name of a tokenizer from the Hugging Face transformers library
        output_dim: int,
        max_length: int = 1024,
        project_out: bool = False,
    ):
        super().__init__(output_dim, output_dim, project_out=project_out)

        from transformers import AutoTokenizer

        # Suppress logging from transformers
        previous_level = logging.root.manager.disable
        logging.disable(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            finally:
                logging.disable(previous_level)

        self.max_length = max_length

        self.token_embedder = nn.Embedding(len(self.tokenizer), output_dim)

    def forward(
        self, texts: tp.List[str], device: tp.Union[torch.device, str]
    ) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        self.proj_out.to(device)

        encoded = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device).to(torch.bool)

        embeddings = self.token_embedder(input_ids)

        embeddings = self.proj_out(embeddings)

        embeddings = embeddings * attention_mask.unsqueeze(-1).float()

        return embeddings, attention_mask


class PretransformConditioner(Conditioner):
    """
    A conditioner that uses a pretransform's encoder for conditioning

    Args:
        pretransform: an instantiated pretransform to use for conditioning
        output_dim: the dimension of the output embeddings
    """

    def __init__(self, pretransform: Pretransform, output_dim: int):
        super().__init__(pretransform.encoded_channels, output_dim)

        self.pretransform = pretransform

    def forward(
        self,
        audio: tp.Union[torch.Tensor, tp.List[torch.Tensor], tp.Tuple[torch.Tensor]],
        device: tp.Union[torch.device, str],
    ) -> tp.Tuple[torch.Tensor, torch.Tensor]:

        self.pretransform.to(device)
        self.proj_out.to(device)

        if isinstance(audio, list) or isinstance(audio, tuple):
            audio = torch.cat(audio, dim=0)

        # Convert audio to pretransform input channels
        audio = set_audio_channels(audio, self.pretransform.io_channels)

        latents = self.pretransform.encode(audio)

        latents = self.proj_out(latents)

        return [
            latents,
            torch.ones(latents.shape[0], latents.shape[2]).to(latents.device),
        ]


class MultiConditioner(nn.Module):
    """
    A module that applies multiple conditioners to an input dictionary based on the keys

    Args:
        conditioners: a dictionary of conditioners with keys corresponding to the keys of the conditioning input dictionary (e.g. "prompt")
        default_keys: a dictionary of default keys to use if the key is not in the input dictionary (e.g. {"prompt_t5": "prompt"})
    """

    def __init__(
        self,
        conditioners: tp.Dict[str, Conditioner],
        default_keys: tp.Dict[str, str] = {},
    ):
        super().__init__()

        self.conditioners = nn.ModuleDict(conditioners)
        self.default_keys = default_keys

    def forward(
        self,
        batch_metadata: tp.List[tp.Dict[str, tp.Any]],
        device: tp.Union[torch.device, str],
    ) -> tp.Dict[str, tp.Any]:
        output = {}

        for key, conditioner in self.conditioners.items():
            condition_key = key

            conditioner_inputs = []

            for x in batch_metadata:
                if condition_key not in x:
                    if condition_key in self.default_keys:
                        condition_key = self.default_keys[condition_key]
                    else:
                        raise ValueError(
                            f"Conditioner key {condition_key} not found in batch metadata"
                        )

                # Unwrap the condition info if it's a single-element list or tuple, this is to support collation functions that wrap everything in a list
                if (
                    isinstance(x[condition_key], list)
                    or isinstance(x[condition_key], tuple)
                    and len(x[condition_key]) == 1
                ):
                    conditioner_input = x[condition_key][0]

                else:
                    conditioner_input = x[condition_key]

                conditioner_inputs.append(conditioner_input)

            output[key] = conditioner(conditioner_inputs, device)

        return output


def create_multi_conditioner_from_conditioning_config(
    config: tp.Dict[str, tp.Any],
) -> MultiConditioner:
    """
    Create a MultiConditioner from a conditioning config dictionary

    Args:
        config: the conditioning config dictionary
        device: the device to put the conditioners on
    """
    conditioners = {}
    cond_dim = config["cond_dim"]

    default_keys = config.get("default_keys", {})

    for conditioner_info in config["configs"]:
        id = conditioner_info["id"]

        conditioner_type = conditioner_info["type"]

        conditioner_config = {"output_dim": cond_dim}

        conditioner_config.update(conditioner_info["config"])

        if conditioner_type == "t5":
            conditioners[id] = T5Conditioner(**conditioner_config)
        elif conditioner_type == "int":
            conditioners[id] = IntConditioner(**conditioner_config)
        elif conditioner_type == "number":
            conditioners[id] = NumberConditioner(**conditioner_config)
        elif conditioner_type == "phoneme":
            conditioners[id] = PhonemeConditioner(**conditioner_config)
        elif conditioner_type == "lut":
            conditioners[id] = TokenizerLUTConditioner(**conditioner_config)
        elif conditioner_type == "avclip_frame":
            conditioners[id] = AVClipFrameConditioner(**conditioner_config)
        elif conditioner_type == "pretransform":
            sample_rate = conditioner_config.pop("sample_rate", None)
            assert (
                sample_rate is not None
            ), "Sample rate must be specified for pretransform conditioners"

            pretransform = create_pretransform_from_config(
                conditioner_config.pop("pretransform_config"), sample_rate=sample_rate
            )

            if conditioner_config.get("pretransform_ckpt_path", None) is not None:
                pretransform.load_state_dict(
                    load_ckpt_state_dict(
                        conditioner_config.pop("pretransform_ckpt_path")
                    )
                )

            conditioners[id] = PretransformConditioner(
                pretransform, **conditioner_config
            )
        else:
            raise ValueError(f"Unknown conditioner type: {conditioner_type}")

    return MultiConditioner(conditioners, default_keys=default_keys)
