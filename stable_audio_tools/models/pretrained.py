import json
import time

from .factory import create_model_from_config
from .utils import load_ckpt_state_dict, load_ckpt_state_dict_modified


from huggingface_hub import hf_hub_download

def get_pretrained_model(name: str):
    
    model_config_path = hf_hub_download(name, filename="model_config.json", repo_type='model')

    with open(model_config_path) as f:
        model_config = json.load(f)

    model = create_model_from_config(model_config)

    # Try to download the model.safetensors file first, if it doesn't exist, download the model.ckpt file
    try:
        model_ckpt_path = hf_hub_download(name, filename="model.safetensors", repo_type='model')
    except Exception as e:
        model_ckpt_path = hf_hub_download(name, filename="model.ckpt", repo_type='model')

    model.load_state_dict(load_ckpt_state_dict(model_ckpt_path))

    return model, model_config

def get_pretrained_model_local(model_config_path: str, model_ckpt_path: str):
    print("Starting model load...")
    t0 = time.time()

    with open(model_config_path) as f:
        model_config = json.load(f)
    print(f"Config loaded in {time.time() - t0:.2f}s")

    t1 = time.time()
    model = create_model_from_config(model_config)
    print(f"Model created in {time.time() - t1:.2f}s")

    t2 = time.time()
    state_dict = load_ckpt_state_dict_modified(model_ckpt_path)
    print(f"State dict loaded in {time.time() - t2:.2f}s")

    t3 = time.time()
    model.load_state_dict(state_dict)
    print(f"State dict applied in {time.time() - t3:.2f}s")

    print(f"Total loading time: {time.time() - t0:.2f}s")
    return model, model_config
