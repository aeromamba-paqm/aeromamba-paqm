from email import parser
import sox
import os
import sys
import argparse
from multiprocessing import Pool


def resample_subdir(data_dir, data_subdir, out_dir, target_sr, flexible=False):
    """_summary_

    Args:
        data_dir (_type_): _description_
        data_subdir (_type_): _description_
        out_dir (_type_): _description_
        target_sr (_type_): _description_
        flexible (bool, optional): If True, changes the filename of the output as {filename}_{target_sr}.wav. Defaults to False.
    """
    print(f'resampling {data_subdir}')
    tfm = sox.Transformer()
    tfm.set_output_format(rate=target_sr)
    out_sub_dir = os.path.join(out_dir, data_subdir)
    if not os.path.isdir(out_sub_dir):
        os.makedirs(out_sub_dir)
    for file in os.listdir(os.path.join(data_dir, data_subdir)):
        in_path = os.path.join(data_dir, data_subdir, file) 
        
        out_filename = file
        if flexible:
            filename, ext = os.path.splitext(file)
            out_filename = f'{filename}_{target_sr}{ext}' 
        
        out_path = os.path.join(out_sub_dir, out_filename)
        
        if os.path.isfile(out_path):
            print(f'{out_path} already exists.')
        elif not file.lower().endswith('.wav'):
            print(f'{in_path}: invalid file type.')
        else:
            success = tfm.build_file(input_filepath=in_path, output_filepath=out_path)
            if success:
                print(f'Succesfully saved {in_path} to {out_path}')


def resample_data(data_dir, out_dir, target_sr, flexible=False):
    with Pool() as p:
        p.starmap(resample_subdir,
                  [(data_dir, data_subdir, out_dir, target_sr, flexible) for data_subdir in os.listdir(data_dir)])

def resample_data_multiple_sr(data_dir, out_dir, target_srs):
    for target_sr in target_srs:
        out_dir_sr = os.path.join(out_dir, str(target_sr))
        resample_data(data_dir, out_dir_sr, target_sr, flexible=True)

def parse_args():
    parser = argparse.ArgumentParser(description='Resample data.')
    parser.add_argument('--data_dir', help='directory containing source files')
    parser.add_argument('--out_dir', help='directory to write target files')
    parser.add_argument('--target_sr', type=int, nargs='+', help='one or more target sample rates')
    parser.add_argument('--flexible', action='store_true', help='if true, appends target sample rate to filename')
    return parser.parse_args()

"""Usage: python data_prep/resample_data.py --data_dir <path for source data> --out_dir <path for target data> --target_sr <target sample rates> --flexible <true or false>"""
def main():
    args = parse_args()
    print(args)
    if len(args.target_sr) > 1:
        print(f"Multiple rates detected: {args.target_sr}. Switching to flexible mode.")
        resample_data_multiple_sr(args.data_dir, args.out_dir, args.target_sr)
    else:
        sr = args.target_sr[0]
        resample_data(args.data_dir, args.out_dir, sr, flexible=args.flexible)
        print(f'Done resampling to target rate {sr}.')
if __name__ == '__main__':
    main()