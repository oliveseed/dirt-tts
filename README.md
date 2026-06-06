# nanoTTS
This project is meant to be like nanoGPT but for text-to-speech. It follows a similar paradigm to Parakeet/Dia but it is scaled down and optimized to be trained from scratch in <1 day on a single GPU.

Try the trained model:
<p>
<a href="https://huggingface.co/spaces/ouasdg/nanoTTS"><img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-lg-dark.svg" alt="Space on HuggingFace" height=38></a>
</p>
It might take a minute for the demo to load because it goes to sleep when inactive.

## Overview

The nanoTTS model is an encoder-decoder Transformer that predicts discrete speech tokens from text. It generates one autoregressive stream of acoustic tokens, then decodes those tokens back to 16 kHz mono audio with Stable Codec.

Specifically:
- Encodes text prompts with a frozen LLaMA token embedding and trainable text encoder.
- Decoder predicts Stable Codec speech tokens autoregressively, using classifier-free guidance during generation by running conditional and unconditional batches together.
- Decodes generated tokens with the `stable-codec-speech-16k` codec using the `1x46656_400bps` posthoc bottleneck.

## Setup

### 1. Requirements

- Python>=3.10
- CUDA toolkit
- NVIDIA GPU, Ampere or newer

### 2. Environment setup

```bash
python3 -m venv nanotts
cd nanotts
source bin/activate
```

### 3. Installation

```bash
git clone https://github.com/oliveseed/nanoTTS.git
cd nanoTTS
pip install -r requirements.txt
pip install -U flash-attn --no-build-isolation
```

### 4. Download audio tokenizer checkpoint

The project uses Stable Codec for audio token decoding. Download the `stable-codec-speech-16k` [weights](https://huggingface.co/stabilityai/stable-codec-speech-16k) from Stability AI and place them somewhere accessible, for example under `checkpoints/`.

Then update the codec section in `config.yaml` with the paths to:

- the Stable Codec checkpoint, usually `model.ckpt`
- the Stable Codec model config, usually `model_config.json`

## Data preparation

See [data/README.md](data/README.md)

## Training

After configuring dataset path, launch single-GPU training:
```bash
python3 train.py
```

Evaluation writes generated audio examples to `eval_out/`. Create that directory before evaluation if it does not already exist:

```bash
mkdir -p eval_out
```

## Inference

`model.py` contains the `TTS` wrapper used for generation:

- `TTS.generate(...)` tokenizes a text prompt and generates audio token IDs.
- `TTS.decode(...)` decodes generated Stable Codec tokens to waveform tensors and 16-bit NumPy audio.
- Sampling supports temperature, top-p filtering, classifier-free guidance, and an optional CFG top-k filter.

## References

- [Stable Codec](https://github.com/Stability-AI/stable-codec)
- [Finite Scalar Quantization](https://arxiv.org/abs/2309.15505)
- [nanoGPT](https://github.com/karpathy/nanoGPT)
