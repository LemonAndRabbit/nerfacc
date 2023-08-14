#!/bin/bash

data_root=/data/zhifan/data/nerf_synthetic/
export CUDA_VISIBLE_DEVICES=7
tag=vanilla_adapt_bg_0.2_center_loss_e35k_nodecay_asize

set -x

for scene in Barn Caterpillar Family Ignatius Truck
do
    dir=ckpt_tanks/${tag}
    ${set_cuda} python examples/extract_mesh.py --scene ${scene} --load_path ${dir}/${scene} --mesh_level 0.2
done
