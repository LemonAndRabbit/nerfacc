#!/bin/bash

data_root=/data/zhifan/data/nerf_synthetic/
export CUDA_VISIBLE_DEVICES=7
tag=random_0.1_ndecay_closs_35k_bbox

set -x

for scene in chair drums ficus hotdog lego materials mic ship
do
    dir=ckpt/${tag}
    ${set_cuda} python examples/extract_mesh.py --scene ${scene} --load_path ${dir}/${scene} --mesh_level 0.1
done
