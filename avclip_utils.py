import torch
import torch.distributed as dist
import torchvision
import logging
from omegaconf import OmegaConf

# Assuming synchformer is importable relative to where this file is placed
# Or adjust sys.path as needed before importing
from synchformer.utils.utils import instantiate_from_config

# Configure logging for avclip_utils
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
if not log.hasHandlers():
    log.addHandler(logging.StreamHandler())


class DistributedDataParallel(torch.nn.parallel.DistributedDataParallel):
    """If the `model` object is wrapped in `torch.nn.parallel.DistributedDataParallel` we have
    to use `model.modules` to get access to methods of the model. This wrapper allows
    to avoid using `if ddp: model.module.* else: model.*`. Used during `evaluate_on_sync_w_shifts`.
    """

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def get_model(cfg: OmegaConf, device: str):
    """Load the AVClip model."""
    model = instantiate_from_config(cfg.model)

    # Set feature extractors to non-trainable
    if cfg.model.params.vfeat_extractor.is_trainable is False:
        for params in model.vfeat_extractor.parameters():
            params.requires_grad = False
    if cfg.model.params.afeat_extractor.is_trainable is False:
        for params in model.afeat_extractor.parameters():
            params.requires_grad = False

    model = model.to(device)
    model_without_ddp = model
    if dist.is_initialized():
        log.info("Initializing DistributedDataParallel for AVClip model.")
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(
            model, device_ids=[cfg.training.local_rank]
        )  # Assuming local_rank is in cfg.training
        model_without_ddp = model.module
    else:
        log.info("DDP not initialized. Using single device for AVClip model.")

    return model, model_without_ddp


def get_transforms(cfg: OmegaConf, which_transforms=["train", "test"]):
    """Get data transforms for the model."""
    transforms = {}
    for mode in which_transforms:
        ts_cfg = cfg.get(f"transform_sequence_{mode}", None)
        ts = (
            [lambda x: x]
            if ts_cfg is None
            else [instantiate_from_config(c) for c in ts_cfg]
        )
        transforms[mode] = torchvision.transforms.Compose(ts)
    return transforms


def patch_config(cfg: OmegaConf):
    """Patch the config for compatibility."""
    # the FE ckpts are already in the model ckpt
    if hasattr(cfg.model.params.afeat_extractor, "params") and hasattr(
        cfg.model.params.afeat_extractor.params, "ckpt_path"
    ):
        cfg.model.params.afeat_extractor.params.ckpt_path = None
    if hasattr(cfg.model.params.vfeat_extractor, "params") and hasattr(
        cfg.model.params.vfeat_extractor.params, "ckpt_path"
    ):
        cfg.model.params.vfeat_extractor.params.ckpt_path = None

    # old checkpoints have different names
    if "transformer" in cfg.model.params:
        if hasattr(cfg.model.params.transformer, "target") and isinstance(
            cfg.model.params.transformer.target, str
        ):
            cfg.model.params.transformer.target = (
                cfg.model.params.transformer.target.replace(
                    ".modules.feature_selector.", ".sync_model."
                )
            )
    else:
        log.warning(
            "Config check: 'transformer' key not found in cfg.model.params. Assuming config is for feature extractor or structure differs."
        )
    return cfg


def generate_embedding(model, cfg: OmegaConf, video_tensor, device="cuda"):
    """
    Generate AV-CLIP embedding from a video tensor.

    Args:
        model: AVClip model (already loaded and on the correct device)
        cfg: Model config (OmegaConf object)
        video_tensor: Video tensor of shape [N, H, W, C], normalized to [0, 1].
                      Will be processed to [N, C, H, W] internally.
                      Expected N=250 frames (10s @ 25fps) for standard processing.
        device: Device to run the model on (e.g., 'cuda' or 'cpu')

    Returns:
        embedding: AV-CLIP embedding tensor.
    """
    log.info("Generating AV-CLIP embedding")

    # Ensure the tensor has the right format and type
    # Input is expected as [N, H, W, C], float [0,1]
    # Convert to uint8 [0, 255] and permute for model
    if video_tensor.dtype != torch.uint8:
        video_tensor_uint8 = (video_tensor * 255).byte()  # Use .byte() for uint8
    else:
        video_tensor_uint8 = video_tensor

    if video_tensor_uint8.shape[0] != 250:
        log.warning(
            f"Video tensor being embedded has {video_tensor_uint8.shape[0]} frames, expected 250 (10s). Ensure padding/truncation occurred before calling this."
        )

    # Permute to N, C, H, W for PyTorch conv layers
    video_tensor_nchw = video_tensor_uint8.permute(0, 3, 1, 2)

    # Resize if necessary (model expects 256x256)
    current_h, current_w = video_tensor_nchw.shape[2], video_tensor_nchw.shape[3]
    if current_h != 256 or current_w != 256:
        log.warning(
            f"Resizing video tensor in generate_embedding from {current_h}x{current_w} to 256x256."
        )
        resize_transform = torchvision.transforms.Resize((256, 256), antialias=True)
        # Ensure float for resize transform
        video_tensor_nchw = resize_transform(video_tensor_nchw.float() / 255.0) * 255.0
        video_tensor_nchw = (
            video_tensor_nchw.byte()
        )  # Convert back to uint8 if needed by transforms

    meta = {
        "video": {"fps": [25]},  # Assuming 25 fps as per standard AVClip
        "audio": {"framerate": [16000]},  # Dummy framerate for silent audio
    }

    # Generate dummy audio to match standard 10-second window (10 seconds * 16kHz)
    dummy_audio = torch.zeros(160000)

    item = dict(
        video=video_tensor_nchw,  # Use the NCHW uint8 tensor
        audio=dummy_audio,
        meta=meta,
        path="",  # Not used but expected by transforms
        split="test",  # Assume test mode for transforms
        targets={
            "v_start_i_sec": 0.0,  # Not used but expected
            "offset_sec": 0.0,  # Not used but expected
        },
    )

    # Apply transforms (assuming 'test' transforms are appropriate)
    try:
        transforms = get_transforms(cfg, ["test"])["test"]
        item = transforms(item)
        # Ensure batch is created correctly - check what transforms output
        # If transforms output a dict, collate should work. If it modifies in-place, handle differently.
        batch = torch.utils.data.default_collate([item])  # Create a batch of size 1
    except Exception as e:
        log.error(f"Error applying transforms or collating batch: {e}")
        raise

    # Prepare inputs for the model
    aud = batch["audio"].to(device)
    vid = batch["video"].to(device)

    # Forward pass
    with torch.no_grad():
        # Enable autocast only if on CUDA
        use_autocast = device == "cuda" and torch.cuda.is_available()
        with torch.autocast("cuda", enabled=use_autocast):
            # Check if the model returns one or two values
            output = model(vid, aud)
            if isinstance(output, tuple) and len(output) >= 1:
                embedding = output[0]  # Assume first output is the embedding
            else:
                embedding = output  # Assume the direct output is the embedding

    log.info(f"Generated embedding with shape: {embedding.shape}")
    return embedding
