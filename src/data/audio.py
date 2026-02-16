import math
import torch
import torchaudio
import torch.nn.functional as F

class Audioset:
    def __init__(self, files=None, length=None, stride=None,
                 pad=True, with_path=False, sample_rate=None,
                 channels=None, fixed_n_examples=None):
        self.files = files
        self.length = length  # Samples if sample_rate is set, Seconds if None
        self.stride = stride or length
        self.with_path = with_path
        self.sample_rate = sample_rate
        self.channels = channels
        self.fixed_n_examples = fixed_n_examples

        self.index_map = [] 

        for file, file_length in self.files:
            if self.sample_rate is None:
                sr = torchaudio.info(str(file)).sample_rate
                effective_length = self.length * sr if self.length else None
                effective_stride = self.stride * sr if self.stride else None
            else:
                effective_length = self.length
                effective_stride = self.stride

            if effective_length is None:
                examples = 1
            elif file_length < effective_length:
                examples = 1 if pad else 0
            else:
                examples = int(math.ceil((file_length - effective_length) / effective_stride) + 1)
            
            if self.fixed_n_examples is not None and examples > self.fixed_n_examples:
                examples = self.fixed_n_examples
            
            for chunk_idx in range(examples):
                self.index_map.append((file, chunk_idx))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, index):
        file, chunk_idx = self.index_map[index]

        if self.sample_rate is None:
            info = torchaudio.info(str(file))
            sr = info.sample_rate
            num_frames = int(self.length * sr) if self.length else -1
            offset = int(self.stride * chunk_idx * sr)
        else:
            sr = self.sample_rate 
            num_frames = self.length or -1
            offset = self.stride * chunk_idx
            
        out, loaded_sr = torchaudio.load(str(file), frame_offset=offset, num_frames=num_frames)

        # mono conversion
        if out.shape[0] != self.channels:
            out = torch.mean(out, dim=0, keepdim=True)
        
        # padding
        if num_frames > 0:
            out = F.pad(out, (0, num_frames - out.shape[-1]))
        
        final_sr = loaded_sr if self.sample_rate is None else self.sample_rate

        if self.with_path:
            return (out, final_sr), file
        else:
            return out, final_sr