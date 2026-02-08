# $\textrm{AEROMamba}_{\textrm{P}}$

<p align="center">
<img width="250" height="250" alt="icon_tp" src="https://github.com/user-attachments/assets/09c8be4c-d782-442c-bec2-b652dccfae7d" />
</p>

This work is currently subject to review by the Journal of the Audio Engineering Society.

## About 
Official PyTorch implementation of 

**Efficient Audio Enhancement with a Differentiable Psychoacoustic Loss**

whose demo is available in our [Webpage](https://aeromamba-paqm.github.io/). $\textrm{AEROMamba}_{\textrm{P}}$ is closely related to our previous model [AEROMamba](https://github.com/aeromamba-super-resolution/aeromamba), which is a particular case that does not use PAQM loss.

## Installation

Requirements:
- Python 3.10.0
- Pytorch 1.12.1
- CUDA 11.3

Instructions:

- **Recommended**: Using Anaconda or Miniconda, run `conda env create -f environment.yml -p /home/user/Anaconda3/envs/env_name`
- Run `pip install -r requirements.txt`

Make sure to unzip the contents of [Mamba](https://github.com/state-spaces/mamba/archive/refs/tags/v1.1.3.post1.zip) (the mamba folder) inside `aeromamba-paqm/src/models/` . Some dependecy warnings can be shown in the output, but the installation succeeds.

If there is any error in the previous step, install manually the required libs. For PyTorch/CUDA and Mamba, manual installation is done through 

- `CAUSAL_CONV1D_FORCE_BUILD=TRUE CAUSAL_CONV1D_SKIP_CUDA_BUILD=TRUE CAUSAL_CONV1D_FORCE_CXX11_ABI=TRUE pip install causal_conv1d==1.1.2.post1`
- `CAUSAL_CONV1D_FORCE_BUILD=TRUE CAUSAL_CONV1D_SKIP_CUDA_BUILD=TRUE CAUSAL_CONV1D_FORCE_CXX11_ABI=TRUE pip install mamba-ssm==1.1.3.post1`
- `conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch`

### PAQM 

The PAQM loss function is implemented as a standalone package in [torchpaqm](https://github.com/bvm810/torchpaqm). In the case of $\textrm{AEROMamba}_{\textrm{P}}$, we use it directly via the `src/paqm` folder. Check the main repo to keep up with updates in the module!

-  **Important:** sometimes, in the first epochs of training with PAQM, the backward propagation can throw an error. This is treated by our code, as long as the anomaly detection is set to `True`. Just keep
  training anyways.
  

### ViSQOL

We did not use ViSQOL for training and validation, but if you want to, see [AERO](https://github.com/slp-rl/aero) for instructions. 

## Datasets

### Download data

For popular music we use the mixture tracks of [MUSDB18-HQ](https://sigsep.github.io/datasets/musdb.html#musdb18-hq-uncompressed-wav) dataset.

For piano music, we collected a private dataset from CDs whose metadata are described in our [Webpage](https://aeromamba-super-resolution.github.io/).

### Resample data

Data are a collection of high/low resolution pairs. Corresponding high and low resolution signals should be in different folders, eg: hr_dataset and lr_dataset. 

To downsample once to, for example, a target 11.025 kHz, from the original 44.1 kHz.

`python data_prep/resample_data.py --data_dir <path for 44.1 kHz data> --out_dir <path for 11.025 kHz data> --target_sr 11025`

### Create egs files

For each low and high resolution pair, one should create "egs files" twice: for low and high resolution.  
`create_meta_files.py` creates a pair of train and val "egs files", each under its respective folder.
Each "egs file" contains meta information about the signals: paths and signal lengths.

`python data_prep/create_meta_files.py <path for 11.025 kHz data> egs/musdb/ lr` 

`python data_prep/create_meta_files.py <path for 44.1 kHz data> egs/musdb/ hr`

The default configuration assume that the dataset folder is organized as 'datasets/dataset_name/partition', where 'partition' can be train, val, or test. We provide a sample of egs folder and files to support the users.

### Generate MP3 files 

To generate MP3 files run  `encode_all.sh` and to transform from MP3 to WAV (for training and inference) use `mp3_to_wav.sh`. Both of them are located inside the `data_prep`
folder.

## Train

You can run $\textrm{AEROMamba}\_{\textrm{P}}$ or $\textrm{AEROMamba}\_{\textrm{P}\bar{\textrm{S}}}$ depending on the `maing_config` files located in the `conf` folder. The difference between them is whether the PAQM loss is present or absent. The experiment
.yaml files control the gamma factor that weights the PAQM loss.

Run `train.py` with `dset` and `experiment` parameters, or set the default values in main_config.yaml file.  

`
python train.py dset=<dset-name> experiment=<experiment-name>
`

To train with multiple GPUs, run with parameter `ddp=true`. e.g.
`
python train.py dset=<dset-name> experiment=<experiment-name> ddp=true
`

## Test (on whole dataset)

`
python test.py dset=<dset-name> experiment=<experiment-name>
`

## Inference

### Single sample

`
python predict.py dset=<dset-name> experiment=<experiment-name> +filename=<absolute path to input file> +output=<absolute path to output directory>
`

### Multiple samples

`
bash predict_batch.sh <input_folder> <output_folder>
`

We also provide predict_with_ola.py to predict large files that do not fit in the GPU, without the need for segmentation, using Overlap-and-Add. The original predict.py is also capable of joining predicted segments, but its naïve method causes clicks. 

`
python predict_batch_with_ola.py dset=<dset-name> experiment=<experiment-name> +folder_path=<absolute path to input folder> +output=<absolute path to output directory>
`
### Checkpoints

To use pre-trained models for MUSDB18-HQ or PianoEval data (both wav and mp3 experiments), one can download checkpoints from [here](https://drive.google.com/drive/folders/1FKOagtfJlqx05zdc1rGZq9rBE44x1kAQ?usp=sharing).

To link to checkpoint when testing or predicting, override/set path under `checkpoint_file:<path>` in `conf/main_config.yaml.` e.g.

`
python test.py dset=<dset-name> experiment=<experiment-name> +checkpoint_file=<path to checkpoint.th file>
`

Alternatively, check that the checkpoint file is in its corresponding output folder:  
For each low to high resolution setting, hydra creates a folder under `outputs/<dset-name>/<experiment-name>`

Make sure that `restart: false` in `conf/main_config.yaml`

      
