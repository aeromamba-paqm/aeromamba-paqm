import sox
import os
import sys
import argparse
import glob
import torchaudio
from collections import namedtuple
import json
from multiprocessing import Process, Manager
import pathlib

FILE_PATTERN='*.wav'

Info = namedtuple("Info", ["length", "sample_rate", "channels"])


def get_info(path):
    info = torchaudio.info(path)
    if hasattr(info, 'num_frames'):
        # new version of torchaudio
        return Info(info.num_frames, info.sample_rate, info.num_channels)
    else:
        siginfo = info[0]
        return Info(siginfo.length // siginfo.channels, siginfo.rate, siginfo.channels)


def add_subdir_meta(subdir_path, shared_meta, n_samples_limit):
    if n_samples_limit and len(shared_meta) > n_samples_limit:
        return
    print(f'creating meta for {subdir_path}')
    audio_files = glob.glob(os.path.join(subdir_path, FILE_PATTERN))
    for idx, file in enumerate(audio_files):
        try:
            info = get_info(file)
            shared_meta.append((file, info.length))
        except Exception as e:
            print(f"Warning: Could not process file {file}. Error: {e}")

    if n_samples_limit and len(shared_meta) > n_samples_limit:
        # Trim excess from this process if it went over
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
        if n_samples_limit:
            meta = meta[:n_samples_limit]
        return meta

def create_meta(data_dir, n_samples_limit=None):
    root, subdirs, files = next(os.walk(data_dir, topdown=True))

    train_path = None
    valid_path = None
    test_path = None

    # === FIX: Find subdirs by name, not index ===
    for subdir in subdirs:
        if subdir.lower() == 'train':
            train_path = os.path.join(root, subdir)
        elif subdir.lower() in ['valid', 'validation']: # Check for 'valid' or 'validation'
            valid_path = os.path.join(root, subdir)
        elif subdir.lower() == 'test':
            test_path = os.path.join(root, subdir)

    # Error if required paths (train, test) are not found
    if not train_path:
        raise FileNotFoundError(f"Could not find a 'train' subdirectory in {data_dir}")
    if not test_path:
        raise FileNotFoundError(f"Could not find a 'test' subdirectory in {data_dir}")

    # Create meta for train and test (mandatory)
    train_subdirs_paths = [train_path]
    test_subdirs_paths = [test_path]
    train_meta = create_subdirs_meta(train_subdirs_paths, n_samples_limit)
    test_meta = create_subdirs_meta(test_subdirs_paths, n_samples_limit)
    
    # === NEW: Create meta for validation (optional) ===
    valid_meta = []
    if valid_path:
        print(f"Found optional 'valid' directory: {valid_path}")
        valid_subdirs_paths = [valid_path]
        valid_meta = create_subdirs_meta(valid_subdirs_paths, n_samples_limit)
    else:
        print("No 'valid' or 'validation' directory found. Skipping validation set.")

    if n_samples_limit:
            print(f"Train meta count: {len(train_meta)}, Valid meta count: {len(valid_meta)}, Test meta count: {len(test_meta)}")

    # Return all three lists
    return train_meta, valid_meta, test_meta



def parse_args():
    parser = argparse.ArgumentParser(description='Resample data.')
    parser.add_argument('data_dir', help='directory containing source files (e.g., /path/to/my_dataset/)')
    parser.add_argument('target_dir', help='output directory for created json files (e.g., egs/chopin-11-44_mp3)')
    parser.add_argument('json_filename', help='filename for created json (e.g., "lr" or "hr")')
    parser.add_argument('--n_samples_limit', type=int, help='limit number of files')
    return parser.parse_args()



"""
usage: python data_prep/create_meta_files.py <data_dir_path> <target_dir> <json_filename>
e.g.:
python create_meta.py /data/sets/my_lr_dataset /data/json_meta lr
python create_meta.py /data/sets/my_hr_dataset /data/json_meta hr
"""
def main():
    args = parse_args()

    # === FIX: Match output directories to your YAML structure ===
    # 'tr' for train, 'valid' for validation, 'tt' for test
    train_dir = os.path.join(args.target_dir, 'tr')
    valid_dir = os.path.join(args.target_dir, 'valid') 
    test_dir = os.path.join(args.target_dir, 'tt')
    
    os.makedirs(args.target_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    # We only make the valid_dir if we find valid data
    
    # Get all three meta lists
    train_meta, valid_meta, test_meta = create_meta(args.data_dir, args.n_samples_limit)

    # --- Write Train JSON ---
    train_json_object = json.dumps(train_meta, indent=4)
    train_json_path = os.path.join(train_dir, args.json_filename + '.json')
    with open(train_json_path, "w") as train_out:
        train_out.write(train_json_object)
    
    # --- Write Test JSON ---
    test_json_object = json.dumps(test_meta, indent=4)
    test_json_path = os.path.join(test_dir, args.json_filename + '.json')
    with open(test_json_path, "w") as test_out:
        test_out.write(test_json_object)

    print(f'Done creating meta for {args.data_dir}.')
    print(f'Created: {train_json_path}')
    print(f'Created: {test_json_path}')
    
    # --- Write Valid JSON (Optional) ---
    if valid_meta:
        os.makedirs(valid_dir, exist_ok=True) # Now make the dir
        valid_json_object = json.dumps(valid_meta, indent=4)
        valid_json_path = os.path.join(valid_dir, args.json_filename + '.json')
        with open(valid_json_path, "w") as valid_out:
            valid_out.write(valid_json_object)
        print(f'Created: {valid_json_path}')


if __name__ == '__main__':
    main()