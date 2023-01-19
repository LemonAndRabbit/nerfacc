## Usage

This is a simple reference for generating initial mesh and initial MLP weights for the **bridge_nerf** codebase.

## Codebase Set Up

Install the [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) dependency:

```bash
pip install git+https://github.com/NVlabs/tiny-cuda-nn#subdirectory=bindings/torch
```

Then locally install this codebase:
```bash
pip install -e .
```

## Train Instant-NGP and extract mesh

The best setting:
```bash
bash scripts/gen_all_0.2_512.sh
```
The output files will be written into `mesh_train_epoch35000/{scene}`