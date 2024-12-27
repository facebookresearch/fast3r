<div align="center">

# Fast3R - Towards 3D Reconstruction of 1000+ Images in a Single Pass


<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
<!-- [![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020) -->
</div>

## Description

3D Reconstruction of 1000+ Images in a Single Pass.

## Installation

```bash
# clone project
git clone https://github.com/facebookresearch/fast3r
cd fast3r

# create conda environment
conda create -n fast3r python=3.11 cmake=3.14.0
conda activate fast3r

# install PyTorch (adjust cuda version according to your system)
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

# Install PyTorch3D
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# install requirements
pip install -r requirements.txt

# install cuda kernels for RoPE following CroCo v2
cd src/croco/models/curope/
python setup.py build_ext --inplace
cd ../../../..
```

## How to run

Train model with default configuration

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 data.batch_size=64
```

## Citation

```
@misc{yang2024fast3r,
      title={Fast3R: 3D Reconstruction of 1000+ Images in a Single Pass}, 
      author={Jianing Yang and Alexander Sax and Kevin J. Liang and Mikael Henaff and Hao Tang and Ang Cao and Joyce Chai and Franziska Meier and Matt Feiszli},
      year={2024},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledgement

Fast3R is built upon a foundation of remarkable open-source projects. We deeply appreciate the contributions of these projects and their communities, whose efforts have significantly advanced the field and made this work possible.
