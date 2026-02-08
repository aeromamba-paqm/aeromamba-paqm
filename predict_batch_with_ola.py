import math
import os

import time

import hydra
import torch
import logging
from pathlib import Path
import numpy as np
import torchaudio
from torchaudio.functional import resample

from src.enhance import write
from src.models import modelFactory
from src.model_serializer import SERIALIZE_KEY_MODELS, SERIALIZE_KEY_BEST_STATES, SERIALIZE_KEY_STATE
from src.utils import bold
import soundfile as sf
logger = logging.getLogger(__name__)


def overlap_and_add(chunks, overlap=256, window_len=1024):
    W = window_len
    win_left_side = np.bartlett(2 * overlap)[:overlap]
    win_right_side = np.bartlett(2 * overlap)[overlap:]
    window = np.concatenate((win_left_side, np.ones(W - 2 * overlap), win_right_side))
    left_window = np.concatenate((np.ones(W - overlap), win_right_side))
    right_window = np.concatenate((win_left_side, np.ones(W - overlap)))    
    n_chunks = len(chunks)
    for i in range(n_chunks):
        if i == 0:
            y = (chunks[i].reshape(-1,) * left_window)
        else:
            x_chunk = chunks[i].reshape(-1,)
            if len(x_chunk) < W or i == n_chunks - 1:
                end_pad = W - len(x_chunk)
                x_chunk = np.pad(x_chunk, (0, end_pad), 'constant', constant_values=0)
                x_ola = x_chunk * right_window
            else:
                x_ola = x_chunk * window
            y = np.pad(y, (0, W - overlap), 'constant', constant_values=0)
            x_ola = np.pad(x_ola, (len(y) - len(x_ola), 0), 'constant', constant_values=0)
            y += x_ola
    return y

SEGMENT_DURATION_SEC = 2

def _load_model(args):
    model_name = args.experiment.model
    checkpoint_file = Path(args.checkpoint_file)
    model = modelFactory.get_model(args)['generator']
    package = torch.load(checkpoint_file, 'cpu')
    load_best = args.continue_best
    if load_best:
        logger.info(bold(f'Loading model {model_name} from best state.'))
        model.load_state_dict(
            package[SERIALIZE_KEY_BEST_STATES][SERIALIZE_KEY_MODELS]['generator'][SERIALIZE_KEY_STATE])
    else:
        logger.info(bold(f'Loading model {model_name} from last state.'))
        model.load_state_dict(package[SERIALIZE_KEY_MODELS]['generator'][SERIALIZE_KEY_STATE])

    return model


@hydra.main(config_path="conf", config_name="main_config_ps")  # for latest version of hydra=1.0
def main(args):
    global __file__
    __file__ = hydra.utils.to_absolute_path(__file__)

    print(args)
    model = _load_model(args)
    device = torch.device('cuda')
    model.cuda()
    folder_path = args.folder_path
    for files in os.listdir(folder_path):
        filename = os.path.join(folder_path, files)
        file_basename = Path(filename).stem
        output_dir = args.output
        lr_sig, sr = torchaudio.load(str(filename))
        if lr_sig.shape[1] > 1:
            lr_sig = torch.mean(lr_sig, dim=0, keepdim=True)
        if args.experiment.upsample:
            lr_sig = resample(lr_sig, sr, args.experiment.hr_sr)
            sr = args.experiment.hr_sr

        logger.info(f'lr wav shape: {lr_sig.shape}')

        segment_duration_samples = sr * SEGMENT_DURATION_SEC
        W_hr = 44095 # 44100 samples minus the edge effect samples
        W_lr = 11025
        overlap_hr = 900 #heuristic value
        overlap_lr = overlap_hr // 4
        n_chunks = math.ceil(lr_sig.shape[-1] / (W_lr - overlap_lr))
        logger.info(f'number of chunks: {n_chunks}')


        lr_chunks = []
        for i in range(n_chunks):
            start = i * (W_lr - overlap_lr)
            end = min(start + W_lr, lr_sig.shape[-1])
            lr_chunks.append(lr_sig[:, start:end])
        pr_chunks = []

        model.eval()
        pred_start = time.time()

        with torch.no_grad():
            for i, lr_chunk in enumerate(lr_chunks):
                pr_chunk = model(lr_chunk.unsqueeze(0).to(device)).squeeze(0)
                #remove edge effect samples (only the 4 final samples are distorted)
                pr_chunk = pr_chunk[:, :-5]
                pr_chunks.append(pr_chunk.cpu())

        pred_duration = time.time() - pred_start
        logger.info(f'prediction duration: {pred_duration}')

        pr_ola = overlap_and_add(pr_chunks, overlap=overlap_hr, window_len=W_hr)
        logger.info(f'pr wav shape: {pr_ola.shape}')

        out_filename_ola = os.path.join(output_dir, file_basename + '.wav')
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f'saving to: {out_filename_ola}, with sample_rate: {args.experiment.hr_sr}')

        sf.write(out_filename_ola, pr_ola, args.experiment.hr_sr)
        

"""
Need to add filename and output to args.
Usage: python predict_batch_with_ola.py <dset> <experiment> +folder_path=<path to input folder> +output=<path to output dir>
"""
if __name__ == "__main__":
    main()