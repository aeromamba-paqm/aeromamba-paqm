"""
This code is based on Facebook's HDemucs code: https://github.com/facebookresearch/demucs
"""
import math
import torchaudio
import torch
import random
from torch.nn import functional as F


class Audioset:
    def __init__(self, files=None, length=None, stride=None,
                 pad=True, with_path=False, sample_rate=None,
                 channels=None, fixed_n_examples=None):
        """
        files should be a list [(file, length)]
        """
        self.files = files
        self.num_examples = []
        self.length = length
        self.stride = stride or length
        self.with_path = with_path
        self.sample_rate = sample_rate
        self.channels = channels
        self.fixed_n_examples = fixed_n_examples

        for file, file_length in self.files:
            if length is None:
                examples = 1
            elif file_length < length:
                examples = 1 if pad else 0
            elif pad:
                examples = int(math.ceil((file_length - self.length) / self.stride) + 1)
                if self.fixed_n_examples is not None:
                    if examples > self.fixed_n_examples:
                        examples = self.fixed_n_examples
            else:
                examples = (file_length - self.length) // self.stride + 1
                if self.fixed_n_examples is not None:
                   if examples > self.fixed_n_examples:
                       examples = self.fixed_n_examples
            self.num_examples.append(examples)

    def __len__(self):
        return sum(self.num_examples)

    def __getitem__(self, index):
        for (file, file_samples), examples in zip(self.files, self.num_examples):
            if index >= examples:
                index -= examples
                continue
            num_frames = 0
            offset = 0
            if self.length is not None:
                num_frames = self.length
                offset = self.stride * index
                
            if torchaudio.get_audio_backend() in ['soundfile', 'sox_io']:
                out, sr = torchaudio.load(str(file),
                                          frame_offset=offset,
                                          num_frames=num_frames or -1)
            else:
                out, sr = torchaudio.load(str(file), offset=offset, num_frames=num_frames)


            if sr != self.sample_rate:
                raise RuntimeError(f"Expected {file} to have sample rate of "
                                   f"{self.sample_rate}, but got {sr}")
            if out.shape[0] != self.channels:
                #raise RuntimeError(f"Expected {file} to have shape of "
                #                   f"{self.channels}, but got {out.shape[0]}")
                #print("Normalizing stereo file")
                out = torch.mean(out, dim=0, keepdim=True)
            if num_frames:
                out = F.pad(out, (0, num_frames - out.shape[-1]))
            if self.with_path:
                return out, file
            else:
                return out
