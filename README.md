# Dirt
Dirt is a low budget text-to-speech model that generates highly realistic speech while being architecturally simple and lightweight enough to be trained from scratch on a single GPU.

### Installation
```bash
git clone https://github.com/oliveseed/dirt-tts.git
cd dirt-tts
pip install -r requirements.txt
pip install -U flash-attn --no-build-isolation
```

## Updates
- [x] Create GitHub repo
- [x] Add inference code
- [ ] Add demos
- [ ] Add training code
- [ ] Add pretrained model checkpoints

## References
- https://github.com/Stability-AI/stable-codec
- https://github.com/nari-labs/dia
- https://github.com/stlohrey/dia-finetuning
- https://github.com/karpathy/nanoGPT