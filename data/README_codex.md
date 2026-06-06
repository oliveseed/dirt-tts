# Dirt-TTS Data Preprocessing

This directory contains the data cleanup scripts used before training Dirt-TTS. They are lightweight research scripts, not packaged command-line tools, so expect to edit paths and column names for your local dataset.

## Files

```text
data/
  prepare_data.py          # Filters tokenized parquet data and writes emilia_300_clean.npy
  classifier.py            # Reusable Wav2Vec2 binary classifier module
  train_classifier.py      # Trains a classifier head on labeled audio clips
  lookdata.ipynb           # Notebook for inspecting prepared data
  test_classifier.ipynb    # Notebook for classifier experiments/evaluation
```

## Prepared TTS Dataset Format

`train.py` loads the prepared dataset with `np.load(..., allow_pickle=True)`. Each row should contain:

- `text`: transcript string
- `tokens`: Stable Codec token sequence for the matching audio

The current `prepare_data.py` exports:

```python
df_clean_np = df_merged[["text", "tokens"]].values
np.save("emilia_300_clean.npy", df_clean_np)
```

The TTS training script then loads that file and reads each item as `(text, tokens)`.

## `prepare_data.py`

`prepare_data.py` combines a tokenized speech dataset with speaker-level synthetic-speech scores, filters low-quality examples, and writes `emilia_300_clean.npy`.

The script currently expects two parquet files:

- a main dataset parquet with at least `text`, `tokens`, `speaker`, `phone_count`, and `duration`
- a speaker score parquet with at least `speaker_id` and `prob_mean`

The filtering logic is:

- compute `phone_density = phone_count / duration`
- join examples to speaker scores with `speaker == speaker_id`
- keep speakers with `prob_mean < 0.1`
- keep examples with `5 <= phone_density <= 25`
- save only `text` and `tokens`

Before running, edit the hard-coded parquet paths near the top of `prepare_data.py`.

```bash
cd dirt-tts
python3 data/prepare_data.py
```

By default this writes `emilia_300_clean.npy` in the current working directory. Move it to the path configured in `train.py` or update `TrainingConfig.dataset_path`.

## Synthetic-Speech Classifier

The classifier is a Wav2Vec2 encoder with a small binary classification head. It is intended to identify synthetic or low-quality generated speech so those speakers can be filtered before TTS training.

There are two copies of the model definition:

- `classifier.py` contains the reusable `w2v2_clf` module.
- `train_classifier.py` defines the same model inline and includes the training loop.

`train_classifier.py` expects:

```text
data/emilia_synth_labels.parquet
```

with at least these columns:

- `data`: path or file-like object readable by `torchaudio.load`
- `label`: binary target, where the positive class is synthetic according to your labeling convention
- `speaker_id`: used for speaker-disjoint train/validation splitting

During collation, each audio sample is:

- loaded with `torchaudio`
- mixed down to mono
- resampled to 16 kHz
- cropped or padded to 3 seconds

Run classifier training from the project root:

```bash
cd dirt-tts
python3 data/train_classifier.py
```

The script trains for one epoch, prints validation classification reports, and writes a classifier-head checkpoint named like:

```text
w2v2_clf_head_<step>.pth
```

Only the classifier head is saved:

```python
torch.save(model.classifier.state_dict(), f"w2v2_clf_head_{step}.pth")
```

To reuse it, instantiate `w2v2_clf` with matching settings and load the saved state dict into `model.classifier`.

## Practical Notes

- The scripts assume CUDA by default (`device = "cuda"`).
- The Wav2Vec2 encoder is downloaded from Hugging Face: `facebook/wav2vec2-base`.
- The classifier freezes the Wav2Vec2 encoder by default and trains only the head.
- The preprocessing flow assumes audio has already been tokenized with the same Stable Codec bottleneck used by Dirt-TTS.
- The notebooks are useful for inspection, but the reproducible path is to keep the filtering choices in `prepare_data.py`.
