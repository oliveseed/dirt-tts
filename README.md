# 土 (Dirt)
Dirt-TTS is a text-to-speech model that generates highly realistic speech and can be trained from scratch on a single GPU. During inference, it generates a single sequence of acoustic tokens in a single pass.

Dirt-TTS predicts latent audio codes autoregressively to generate speech in __16 kHz mono__. Audio is tokenized using the `stable-codec-speech-16k` variant of [Stable Codec](https://github.com/Stability-AI/stable-codec), set at the `1x46656_400bps` bottleneck preset. The bottleneck is based on [Finite Scalar Quantization](https://arxiv.org/abs/2309.15505) and allows the codec to encode speech at extremely low bitrates. Since there is only one codebook level and the model does not generate spectrograms, it does not require a vocoder, delay pattern, or hierarchical Transformer.

### Live demo
<p>
<a href="https://huggingface.co/spaces/ouasdg/dirt-tts"><img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-lg-dark.svg" alt="Space on HuggingFace" height=38></a>
</p>
It might take a minute for the demo to load because it goes to sleep when inactive.

## Setup
Recommend Python>=3.10. Install with pip into a virtualenv
```bash
git clone https://github.com/oliveseed/dirt-tts.git
cd dirt-tts
pip install -r requirements.txt
pip install -U flash-attn --no-build-isolation
```
Then, download Stable Codec [weights](https://huggingface.co/stabilityai/stable-codec-speech-16k) and place in `checkpoints/`.

## Training
Configure dataset path and optional WandB logging then run
```bash
python3 train.py
```

## Updates
- [x] Create GitHub repo
- [x] Add inference code
- [x] Add training code
- [x] Add basic live demo
- [ ] Add pretrained model checkpoints
- [ ] Add data preprocessing code

## References
- https://github.com/Stability-AI/stable-codec
- https://github.com/nari-labs/dia
- https://github.com/stlohrey/dia-finetuning
- https://github.com/karpathy/nanoGPT
