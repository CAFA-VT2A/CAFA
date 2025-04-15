<div align="center">

# CAFA - a Controllable Automatic Foley Artist

[Website (TODO)](TODO) |
[Paper](https://arxiv.org/abs/2504.06778) |
[Model](https://huggingface.co/MichaelFinkelson/CAFA-avclip)

</div>

# Introduction

**CAFA (Controllable Automatic Foley Artist)** is a controllable text-video-to-audio model for Foley sound generation. Given a short video and a textual prompt, CAFA generates a synchronized audio waveform that matches both the visual content and the desired semantics described in the prompt. This allows users to modify or override the natural sound of the video by changing the prompt, enabling fine-grained control over the generated audio.

This repository provides the inference tools, pretrained weights, and test results to reproduce our results or build upon our work.
# Examples
## Demo video
[<video src="assets/demo_video.mp4" controls width="100%"></video>
](https://github.com/user-attachments/assets/60e2e8b8-27c3-4060-aef9-2ed2cadee0ad)
## Creative prompts conditioing
| *Typing on typewriter* | *Playing drumkit* |
|------------------------|------------------|
| <video src="assets/our_typing_on_typewriter_drum.mp4" controls width="100%"></video> | <video src="assets/our_playing_drum_kit.mp4" controls width="100%"></video> |

| *Man burping* | *Cattle mooing* |
|---------------|----------------|
| <video src="assets/our_man_burping.mp4" controls width="100%"></video> | <video src="assets/our_cattle_mooing.mp4" controls width="100%"></video> |

| *Dog barking* | *Typing on typewriter* |
|---------------|------------------------|
| <video src="assets/our_dog_barking.mp4" controls width="100%"></video> | <video src="assets/our_typing_on_typewriter_true.mp4" controls width="100%"></video> |
# Installation
1. We recommend working in a fresh enviornment
```bash
git clone https://github.com/finmickey/CAFA.git
cd CAFA

# create env
python -m venv env
source env/bin/activate
```
2. Install preqrequisite if not installed yet
```bash
pip install torch torchvision torchaudio
# Used for downloading the ckpt
git lfs install
``` 
3. Install requirements (using legacy resolver speeds up CUDA dependencies installation, but is optional)
```bash
pip install -r requirements.txt --use-deprecated=legacy-resolver
``` 
4. Download ckpts and config files. Notice that we use the avclip model from [Synchformer](https://github.com/v-iashin/Synchformer), but use a different config.
```bash
mkdir ckpts

wget -O ckpts/avclip.pt https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/sync/sync_models/24-01-04T16-39-21/24-01-04T16-39-21.pt

GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/MichaelFinkelson/CAFA-avclip ckpts/

cd ckpts
git lfs pull
cd ..
``` 
# Inference
An inference script is provided at `demo.py`
```bash
python demo.py --video_path <path_to_video_file.mp4> --prompt "your_prompt"
``` 
The output will be saved at `./output` by default as both `.wav` and combined `.mp4` files.

The computed video embeddings will be cached at `./embeds` and can be reused for faster generation:
```bash
python demo.py --video_path <path_to_video_file.mp4> --embed_path <path_to_embedding_file.npy> --prompt "your_prompt"
``` 

The model supports generation of up to 10 seconds of audio. Videos longer than that will be trimmed to 10 seconds.

## Additional options
Common parameters:
- `--cfg`: Classifier-free guidance scale (default: 7.0)
- `--steps`: Number of diffusion steps (default: 50)
- `--seed`: Random seed (default: 42)
- `--asym_cfg`: Asymmetric CFG scale (default: 0.5)

For all available options:
```bash
python demo.py --help
```
# Outputs
We provide the model's generations of the VGGSound test set at [this huggingface dataset]()
# Citation
```bibtex
@article{benita2025controllableautomaticfoleyartist,
      title={Controllable Automatic Foley Artist}, 
      author={Roi Benita and Michael Finkelson and Tavi Halperin and Gleb Sterkin and Yossi Adi},
      year={2025},
      journal={arXiv preprint arXiv:2504.06778},
      url={https://arxiv.org/abs/2504.06778}, 
}
```
# Acknowledgement
The code is primarily based on [stable-audio-tools](https://github.com/Stability-AI/stable-audio-tools) and [Synchformer](https://github.com/v-iashin/Synchformer).
