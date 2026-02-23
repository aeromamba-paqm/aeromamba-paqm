import os
import argparse
import glob
import torchaudio
import json
from collections import namedtuple
from multiprocessing import Process, Manager

FILE_PATTERN = '*.wav'
Info = namedtuple("Info", ["length", "sample_rate", "channels"])

def get_info(path):
    info = torchaudio.info(path)
    if hasattr(info, 'num_frames'):
        return Info(info.num_frames, info.sample_rate, info.num_channels)
    else:
        siginfo = info[0]
        return Info(siginfo.length // siginfo.channels, siginfo.rate, siginfo.channels)

def add_subdir_meta(subdir_path, shared_meta, n_samples_limit):
    if n_samples_limit and len(shared_meta) > n_samples_limit:
        return
    # Use basename for logging to keep it clean
    print(f'   Scanning: .../{os.path.basename(os.path.dirname(subdir_path))}/{os.path.basename(subdir_path)}')
    
    audio_files = sorted(glob.glob(os.path.join(subdir_path, FILE_PATTERN)))
    for file in audio_files:
        try:
            info = get_info(file)
            shared_meta.append((os.path.abspath(file), info.length))
        except Exception as e:
            print(f"Warning: Could not process {file}: {e}")
    
    if n_samples_limit and len(shared_meta) > n_samples_limit:
        while len(shared_meta) > n_samples_limit:
            shared_meta.pop()

def create_subdirs_meta(subdirs_paths, n_samples_limit):
    with Manager() as manager:
        shared_meta = manager.list()
        processes = []
        for subdir_path in subdirs_paths:
            p = Process(target=add_subdir_meta, args=(subdir_path, shared_meta, n_samples_limit))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        
        meta = list(shared_meta)
        meta.sort()
        return meta[:n_samples_limit] if n_samples_limit else meta

def save_json(data, target_dir, filename):
    os.makedirs(target_dir, exist_ok=True)
    path = os.path.join(target_dir, f"{filename}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    return path

def parse_args():
    parser = argparse.ArgumentParser(description='Create paired JSON metadata for Flexible Audio SR.')
    parser.add_argument('--lr_dir', required=True, help='Dir folder containing SR folders (10000, 15000, etc)')
    parser.add_argument('--hr_dir', required=True, help='Dir folder containing HR splits (train, val, test)')
    parser.add_argument('--target_dir', required=True, help='Where to save the output tr/val/tt json folders')
    parser.add_argument('--n_samples_limit', type=int, help='Limit files per SR variant per split')
    return parser.parse_args()

def main():
    args = parse_args()
    splits = ['tr', 'val', 'tt']
    folder_map = {'tr': 'train', 'val': 'val', 'tt': 'test'}

    # 1. Identify all Sampling Rate folders in the LR Dir
    sr_folders = [d for d in os.listdir(args.lr_dir) 
                  if os.path.isdir(os.path.join(args.lr_dir, d))]
    
    print(f"Detected SR variants in LR dir: {sr_folders}")

    for split_key in splits:
        split_name = folder_map[split_key]
        print(f"\n--- Constructing Split: {split_key} ({split_name}) ---")
        
        lr_accumulated = []
        hr_accumulated = []

        for sr in sr_folders:
            lr_split_path = os.path.join(args.lr_dir, sr, split_name)
            hr_split_path = os.path.join(args.hr_dir, split_name)

            if os.path.exists(lr_split_path) and os.path.exists(hr_split_path):
                lr_meta = create_subdirs_meta([lr_split_path], args.n_samples_limit)
                hr_meta = create_subdirs_meta([hr_split_path], args.n_samples_limit)

                lr_meta.sort()
                hr_meta.sort()

                if len(lr_meta) != len(hr_meta):
                    print(f" ! Warning: Mismatch in {sr}/{split_name}. LR: {len(lr_meta)}, HR: {len(hr_meta)}")
                
                lr_accumulated.extend(lr_meta)
                hr_accumulated.extend(hr_meta)
            else:
                print(f"   Skipping {sr}/{split_name}: Path not found.")

        if lr_accumulated:
            split_target_dir = os.path.join(args.target_dir, split_key)
            save_json(lr_accumulated, split_target_dir, "lr")
            save_json(hr_accumulated, split_target_dir, "hr")
            print(f"Final {split_key} set: {len(lr_accumulated)} pairs saved.")


#usage: python data_prep/create_meta_files.py --lr_dir test_multiple_resample/ --hr_dir probing_datasets/chopin_tinier/ --target_dir egs/test_multiple_resample --n_samples_limit 4
if __name__ == '__main__':
    main()