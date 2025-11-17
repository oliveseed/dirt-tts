import importlib
import random
import torch
import torchaudio
import webdataset as wds

from typing import List
from torchaudio import transforms as T

from stable_audio_tools.data.dataset import (
    S3DatasetConfig, LocalWebDatasetConfig, log_and_continue, audio_decoder, npy_decoder,
    is_valid_sample, collation_fn, AUDIO_KEYS, remove_long_silence,
)
from stable_audio_tools.data.utils import (
    Stereo, Mono, PhaseFlipper, PadCrop_Normalized_T, VolumeNorm,
)

from .Text2Phone.Text2PhoneTokenizer import Text2PhoneTokenizer

class WebDatasetDataLoader():
    def __init__(
        self,
        datasets: List[S3DatasetConfig],
        batch_size,
        sample_size,
        sample_rate=48000,
        num_workers=8,
        epoch_steps=1000,
        random_crop=True,
        force_channels="stereo",
        augment_phase=True,
        remove_silence=True,
        silence_threshold=[0.01, 0.5],
        max_silence_duration=0.2,
        volume_norm=False,
        volume_norm_param=(-16, 2),
        pre_encoded=False,
        resampled_shards=True,
        force_align_text=False,
        **data_loader_kwargs
    ):

        self.datasets = datasets

        self.sample_size = sample_size
        self.sample_rate = sample_rate
        self.random_crop = random_crop
        self.force_channels = force_channels
        self.augment_phase = augment_phase
        self.pre_encoded = pre_encoded
        self.volume_norm = volume_norm
        self.volume_norm_param = volume_norm_param
        self.remove_silence = remove_silence
        self.silence_threshold = silence_threshold
        self.max_silence_duration = max_silence_duration

        self.force_align_text = force_align_text
        self.phonemizer = Text2PhoneTokenizer()

        urls = [dataset.load_data_urls() for dataset in datasets]

        # Flatten the list of lists of URLs
        urls = [url for dataset_urls in urls for url in dataset_urls]

        # Shuffle the urls
        random.shuffle(urls)

        self.dataset = wds.DataPipeline(
            wds.ResampledShards(urls) if resampled_shards else wds.SimpleShardList(urls),
            wds.tarfile_to_samples(handler=log_and_continue),
            wds.decode(audio_decoder, handler=log_and_continue) if not self.pre_encoded else wds.decode(npy_decoder, handler=log_and_continue),
            wds.map(self.wds_preprocess, handler=log_and_continue),
            wds.select(is_valid_sample),
            wds.to_tuple("audio", "json", handler=log_and_continue),
            #wds.shuffle(bufsize=1000, initial=5000),
            wds.batched(batch_size, partial=False, collation_fn=collation_fn),
        )

        if resampled_shards:
            self.dataset = self.dataset.with_epoch(epoch_steps//num_workers if num_workers > 0 else epoch_steps)

        self.data_loader = wds.WebLoader(self.dataset, num_workers=num_workers, **data_loader_kwargs)

    def wds_preprocess(self, sample):

        if self.pre_encoded:
            audio = torch.from_numpy(sample["npy"])
            del sample["npy"]
            sample["__pre_encoded__"] = True
            sample["json"]["padding_mask"] = torch.tensor(sample["json"]["padding_mask"])
        else:
            found_key, rewrite_key = '', ''
            for k, v in sample.items():  # print the all entries in dict
                for akey in AUDIO_KEYS:
                    if k.endswith(akey):
                        # to rename long/weird key with its simpler counterpart
                        found_key, rewrite_key = k, akey
                        break
                if '' != found_key:
                    break
            if '' == found_key:  # got no audio!
                return None  # try returning None to tell WebDataset to skip this one

            audio, in_sr = sample[found_key]
            if in_sr != self.sample_rate:
                resample_tf = T.Resample(in_sr, self.sample_rate)
                audio = resample_tf(audio)

            # Replace the long silence by the short for the mono audios
            if audio.shape[0] == 1 and self.remove_silence:
                audio = remove_long_silence(audio, self.sample_rate, self.silence_threshold, self.max_silence_duration)

            original_length = audio.shape[-1]

            if self.sample_size is not None:
                # Pad/crop and get the relative timestamp
                pad_crop = PadCrop_Normalized_T(
                    self.sample_size, randomize=self.random_crop, sample_rate=self.sample_rate)
                audio, t_start, t_end, seconds_start, seconds_total, padding_mask = pad_crop(
                    audio)
                sample["json"]["seconds_start"] = seconds_start
                sample["json"]["seconds_total"] = seconds_total
                sample["json"]["padding_mask"] = padding_mask
            else:
                t_start, t_end = 0, 1

            start_time = (original_length * t_start) / self.sample_rate
            end_time = (original_length * t_end) / self.sample_rate

            # Check if audio is length zero, initialize to a single zero if so
            if audio.shape[-1] == 0:
                audio = torch.zeros(1, 1)

            # Make the audio stereo and augment by randomly inverting phase
            augs = torch.nn.Sequential(
                Stereo() if self.force_channels == "stereo" else torch.nn.Identity(),
                Mono() if self.force_channels == "mono" else torch.nn.Identity(),
                VolumeNorm(self.volume_norm_param, self.sample_rate) if self.volume_norm else torch.nn.Identity(),
                PhaseFlipper() if self.augment_phase else torch.nn.Identity()
            )

            audio = augs(audio)

            sample["json"]["timestamps"] = (t_start, t_end)

            if found_key != rewrite_key:   # rename long/weird key with its simpler counterpart
                del sample[found_key]

        if "text" in sample["json"]:
            sample["json"]["prompt"] = sample["json"]["text"]

        # Check for custom metadata functions
        for dataset in self.datasets:
            if dataset.custom_metadata_fn is None:
                continue

            if dataset.path in sample["__url__"]:
                custom_metadata = dataset.custom_metadata_fn(sample["json"], audio)
                sample["json"].update(custom_metadata)

        sample["audio"] = audio
        # Add audio to the metadata as well for conditioning
        sample["json"]["audio"] = audio

        if self.force_align_text and self.sample_size is not None:
            # Chunk the original transcriptions according to (start_time, end_time)
            chunked_text_list = []
            for entry in sample["json"]['force_aligned_text']['transcript']:
                word_start = entry['start']
                word_end = entry['end']
                # Check if the word's start or end time falls within the time range
                if (word_start >= start_time and word_start <= end_time) or (word_end >= start_time and word_end <= end_time):
                    chunked_text_list.append(entry['word'])

            chunked_text =  ' '.join(chunked_text_list)
            chunked_phone = self.phonemizer.tokenize(chunked_text)

            sample["json"]["phone"] = chunked_phone
            sample["json"]["aligned_text"] = chunked_text

        return sample

def create_dataloader_from_config(dataset_config, batch_size, sample_size, sample_rate, audio_channels=2, num_workers=4, shuffle = True):
    dataset_type = dataset_config.get("dataset_type", None)
    assert dataset_type is not None, "Dataset type must be specified in dataset config"
    assert dataset_type in ("s3", "wds")

    force_channels = "mono" if audio_channels == 1 else "stereo"

    wds_configs = []
    for wds_config in dataset_config["datasets"]:
        custom_metadata_fn = None
        custom_metadata_module_path = wds_config.get("custom_metadata_module", None)

        if custom_metadata_module_path is not None:
            spec = importlib.util.spec_from_file_location(
                "metadata_module", custom_metadata_module_path)
            metadata_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(metadata_module)
            custom_metadata_fn = metadata_module.get_custom_metadata

        if "s3_path" in wds_config:
            wds_configs.append(S3DatasetConfig(
                id=wds_config["id"],
                s3_path=wds_config["s3_path"],
                custom_metadata_fn=custom_metadata_fn,
                profile=wds_config.get("profile", None),
            ))
        elif "path" in wds_config:
            wds_configs.append(LocalWebDatasetConfig(
                id=wds_config["id"],
                path=wds_config["path"],
                custom_metadata_fn=custom_metadata_fn
            ))

    return WebDatasetDataLoader(
        wds_configs,
        sample_rate=sample_rate,
        sample_size=sample_size,
        batch_size=batch_size,
        remove_silence=dataset_config.get("remove_silence", False),
        silence_threshold=dataset_config.get("silence_threshold", [0.01, 0.5]),
        max_silence_duration=dataset_config.get("max_silence_duration", 0.25),
        random_crop=dataset_config.get("random_crop", True),
        volume_norm=dataset_config.get("volume_norm", False),
        volume_norm_param=dataset_config.get("volume_norm_param", [-16, 2]),
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True,
        force_channels=force_channels,
        epoch_steps=dataset_config.get("epoch_steps", 2000),
        pre_encoded=dataset_config.get("pre_encoded", False),
        resampled_shards=dataset_config.get("resampled_shards", True),
        force_align_text=dataset_config.get("force_align_text", False)
    ).data_loader
