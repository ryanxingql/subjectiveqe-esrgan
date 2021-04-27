# Subjective Quality Enhancement of Compressed Images using ESRGAN

- [Subjective Quality Enhancement of Compressed Images using ESRGAN](#subjective-quality-enhancement-of-compressed-images-using-esrgan)
  - [0. Background](#0-background)
  - [1. Pre-request](#1-pre-request)
    - [1.1. Environment](#11-environment)
    - [1.2. DIV2K dataset](#12-div2k-dataset)
  - [2. Pre-train generator (RRDBNet)](#2-pre-train-generator-rrdbnet)
  - [3. Train](#3-train)
  - [4. Test](#4-test)
  - [5. License & Citation](#5-license--citation)

:rocket: **Update** (21/4/27): We open-source the improved ESRGAN and some commonly-used compared approaches at [PowerQE](https://github.com/RyanXingQL/PowerQE).

## 0. Background

PyTorch implementation of [ESRGAN](https://github.com/xinntao/ESRGAN) for compressed image subjective quality enhancement.

**Note**: The network structure, dataset and training method are different from those in the original paper.

![demo](https://user-images.githubusercontent.com/34084019/105740844-8d83c400-5f74-11eb-9e95-8c0263e32d96.png)

![demo](https://user-images.githubusercontent.com/34084019/105740851-91afe180-5f74-11eb-958a-1378adf8003b.png)

Feel free to contact: <ryanxingql@gmail.com>.

## 1. Pre-request

### 1.1. Environment

- UBUNTU 20.04 LTS
- CUDA 10.1
- PYTORCH 1.6
- packages: TQDM, LMDB, PYYAML, OPENCV-PYTHON, SCIKIT-IMAGE, TENSORBOARDX, LPIPS

Suppose that you have installed CUDA 10.1, then:

```bash
git clone --depth=1 https://github.com/RyanXingQL/SubjectiveQE-ESRGAN
cd SubjectiveQE-ESRGAN/
conda create -n esrgan python=3.7 -y
conda activate esrgan
python -m pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install tqdm lmdb pyyaml opencv-python scikit-image tensorboard lpips
```

### 1.2. DIV2K dataset

<details>
<summary><b>Download dataset</b></summary>
<p>

**Download** the [[DIV2K_train_HR.zip]](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip), [[DIV2K_valid_HR.zip]](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip) and compression toolbox at
 [[Releases]](https://github.com/RyanXingQL/SubjectiveQE-ESRGAN/releases) or [[百度网盘 (sgan)]](https://pan.baidu.com/s/1GQheI6c9lNVvz-437eAZKA).

**Unzip** and **put** folders as:

```tex
DIV2K/
├── compress_ai/
├── DIV2K_train_HR/
│   ├── 0001.png
│   ├── ...
│   └── 0800.png
└── DIV2K_valid_HR/
    ├── 0801.png
    ├── ...
    └── 0900.png
```

</p>
</details>

<details>
<summary><b>Compress images</b></summary>
<p>

**Compress** these PNGs under HEVC all-intra (ai) mode:

```bash
cd compress_ai/
chmod +x TAppEncoderStatic
python convert_n_compress.py
```

We will get:

```tex
DIV2K/
├── compress_ai/
├── DIV2K_train_HR/  # PNG
├── DIV2K_train_HR_crop/  # cropped PNG
├── DIV2K_train_HR_ycbcr/  # PNG -> YUV
├── DIV2K_train_HR_ycbcr_intra/  # compressed YUV
├── DIV2K_train_HR_intra/  # compressed YUV -> PNG
├── DIV2K_valid_HR/
├── DIV2K_valid_HR_crop/
├── DIV2K_valid_HR_ycbcr/
├── DIV2K_valid_HR_ycbcr_intra
└── DIV2K_valid_HR_intra/
```

**Note**:

1. We crop PNG so that height (and also width) is a multiple of the minimum CU size (8). It's required by the HM16.5 codec.
2. We first convert PNG to YCbCr YUV 444P, and convert back after compression. That's because HM16.5 requires YUV as input.

</p>
</details>

<details>
<summary><b>Generate LMDB</b></summary>
<p>

Finally, we generate **LMDB** for training data. Edit `dataset/root` at `option_rrdbnet_div2k.yml`, and run:

```bash
python create_lmdb_div2k.py --opt_path option_rrdbnet_div2k.yml
```

We can then access DIV2K dataset at our repository root.

```tex
SubjectiveQE-ESRGAN/
├── data/
│   └── DIV2K/
│       ├── div2k_train_gt.lmdb/
│       ├── div2k_train_lq_intra_qp37.lmdb/
│       └── ...
└── ...
```

</p>
</details>

## 2. Pre-train generator (RRDBNet)

```bash
# single-gpu
$ CUDA_VISIBLE_DEVICES=0 python train_rrdbnet.py --opt_path option_rrdbnet_div2k.yml  

# multi-gpu
# change num_worker_per_gpu and batch_size_per_gpu at YAML!
$ CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=22222 train_rrdbnet.py --opt_path option_rrdbnet_div2k.yml
```

<details>
<summary><b>Visualization</b></summary>

```bash
tensorboard --logdir=exp/TrainRRDBNetDIV2KIntraQP37 --port=7777
```

![vis](https://user-images.githubusercontent.com/34084019/105740895-9c6a7680-5f74-11eb-833d-a249d9fbfaad.png)

</p>
</details>

**Note**:

- For the original settings at YAML and single-gpu training, ~20338 MiB GPU memory is needed. You can change `batch_size_per_gpu` and network settings at YAML.
- `dataset/num_worker_per_gpu` * [your assigned gpu number] should better be smaller than your cpu core number. For example, we have 40 cpu cores in total and use two gpus, then we should better let `dataset/num_worker_per_gpu` < [40/2=20], e.g., 16.

We have prepared trained models at [[Releases]](https://github.com/RyanXingQL/SubjectiveQE-ESRGAN/releases) and [[百度网盘 (sgan)]](https://pan.baidu.com/s/1GQheI6c9lNVvz-437eAZKA).

## 3. Train

```bash
# single-gpu
$ CUDA_VISIBLE_DEVICES=0 python train_esrgan.py --opt_path option_esrgan_div2k.yml  

# multi-gpu
# change num_worker_per_gpu and batch_size_per_gpu at YAML!
$ CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=22222 train_esrgan.py --opt_path option_esrgan_div2k.yml
```

<details>
<summary><b>Visualization</b></summary>
<p>

```bash
tensorboard --logdir=exp/TrainESRGANDIV2KIntraQP37 --port=7777
```

![vis](https://user-images.githubusercontent.com/34084019/105740877-97a5c280-5f74-11eb-92a7-23ed142b6494.png)

According to the validation curve, we select the model at iteration 130k.

![vis](https://user-images.githubusercontent.com/34084019/105740887-9a081c80-5f74-11eb-9d08-e9287349ed41.png)

Enhanced patches seem much better than the input compressed patches.

</p>
</details>

Results are logged at `exp/TrainESRGANDIV2KIntraQP37/log.log`.

**Note**:

- For the original settings at YAML and single-gpu training, ~21424 MiB GPU memory is needed. You can change `batch_size_per_gpu` and network settings at YAML.
- `dataset/num_worker_per_gpu` * [your assigned gpu number] should better be smaller than your cpu core number. For example, we have 40 cpu cores in total and use two gpus, then we should better let `dataset/num_worker_per_gpu` < [40/2=20], e.g., 16.

We have prepared trained models at [[Releases]](https://github.com/RyanXingQL/SubjectiveQE-ESRGAN/releases) and [[百度网盘 (sgan)]](https://pan.baidu.com/s/1GQheI6c9lNVvz-437eAZKA).

`events.out.tfevents` is also there. It may be too large to download (1.4GB).

## 4. Test

```bash
# only single-gpu is supported
$ CUDA_VISIBLE_DEVICES=0 python test_esrgan.py --opt_path option_esrgan_div2k.yml
```

Result (LPIPS): 0.138 (original) -> 0.080 (enhanced).

Results are logged at `exp/TrainESRGANDIV2KIntraQP37/log_val.log`. Enhanced images are stored at `exp/TrainESRGANDIV2KIntraQP37/img_val/`.

## 5. License & Citation

You can **use, redistribute, and adapt** the material for **non-commercial purposes**, as long as you give appropriate credit by **citing the following paper** and **indicating any changes** that you've made.

```tex
@incollection{Wang_2019,
	doi = {10.1007/978-3-030-11021-5_5},
	url = {https://doi.org/10.1007%2F978-3-030-11021-5_5},
	year = 2019,
	publisher = {Springer International Publishing},
	pages = {63--79},
	author = {Xintao Wang and Ke Yu and Shixiang Wu and Jinjin Gu and Yihao Liu and Chao Dong and Yu Qiao and Chen Change Loy},
	title = {{ESRGAN}: Enhanced Super-Resolution Generative Adversarial Networks},
	booktitle = {Lecture Notes in Computer Science}
}
```

If you find this repository helpful, you may cite:

```tex
@misc{ESRGAN_xing_2020,
  author = {Qunliang Xing},
  title = {Subjective Quality Enhancement of Compressed Images using ESRGAN},
  howpublished = "\url{https://github.com/RyanXingQL/SubjectiveQE-ESRGAN}",
  year = {2020}, 
  note = "[Online; accessed 11-April-2021]"
}
```
