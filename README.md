<p>
  <!-- pypi-strip -->
  <picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://user-images.githubusercontent.com/3310961/199083722-881a2372-62c1-4255-8521-31a95a721851.png" />
  <source media="(prefers-color-scheme: light)" srcset="https://user-images.githubusercontent.com/3310961/199084143-0d63eb40-3f35-48d2-a9d5-78d1d60b7d66.png" />
  <!-- /pypi-strip -->
  <img alt="nerfacc logo" src="https://user-images.githubusercontent.com/3310961/199084143-0d63eb40-3f35-48d2-a9d5-78d1d60b7d66.png" width="350px" />
  <!-- pypi-strip -->
  </picture>
  <!-- /pypi-strip -->
</p>

[![Core Tests.](https://github.com/KAIR-BAIR/nerfacc/actions/workflows/code_checks.yml/badge.svg)](https://github.com/KAIR-BAIR/nerfacc/actions/workflows/code_checks.yml)
[![Documentation Status](https://readthedocs.com/projects/plenoptix-nerfacc/badge/?version=latest)](https://www.nerfacc.com/en/latest/?badge=latest)
[![Downloads](https://pepy.tech/badge/nerfacc)](https://pepy.tech/project/nerfacc)

https://www.nerfacc.com/

[News] 2023/04/04. If you were using `nerfacc <= 0.3.5` and would like to migrate to our latest version (`nerfacc >= 0.5.0`), Please check the [CHANGELOG](CHANGELOG.md) on how to migrate.

NerfAcc is a PyTorch Nerf acceleration toolbox for both training and inference. It focus on
efficient sampling in the volumetric rendering pipeline of radiance fields, which is 
universal and plug-and-play for most of the NeRFs.
With minimal modifications to the existing codebases, Nerfacc provides significant speedups 
in training various recent NeRF papers.
**And it is pure Python interface with flexible APIs!**

![Teaser](/docs/source/_static/images/teaser.jpg?raw=true)

## Installation

**Dependence**: Please install [Pytorch](https://pytorch.org/get-started/locally/) first.

The easist way is to install from PyPI. In this way it will build the CUDA code **on the first run** (JIT).
```
pip install nerfacc
```

Or install from source. In this way it will build the CUDA code during installation.
```
pip install git+https://github.com/KAIR-BAIR/nerfacc.git
```

We also provide pre-built wheels covering major combinations of Pytorch + CUDA supported by [official Pytorch](https://pytorch.org/get-started/previous-versions/).

```
# e.g., torch 1.13.0 + cu117
pip install nerfacc -f https://nerfacc-bucket.s3.us-west-2.amazonaws.com/whl/torch-1.13.0_cu117.html
```

| Windows & Linux | `cu113` | `cu115` | `cu116` | `cu117` | `cu118` |
|-----------------|---------|---------|---------|---------|---------|
| torch 1.11.0    | ✅      | ✅      |         |         |         |
| torch 1.12.0    | ✅      |         | ✅      |         |         |
| torch 1.13.0    |         |         | ✅      | ✅      |         |
| torch 2.0.0     |         |         |         | ✅      | ✅      |

For previous version of nerfacc, please check [here](https://nerfacc-bucket.s3.us-west-2.amazonaws.com/whl/index.html) on the supported pre-built wheels.

## Usage

1. To finish stage1 training for MarryRecon:

```
bash scripts/run_all_stage1_12_18_0.1.sh
```

2. To obtain mesh initialization:
```
bash scripts/extract_all_arachitecture_search_0.2.sh
```