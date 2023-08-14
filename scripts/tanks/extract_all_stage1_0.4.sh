#!/bin/bash

data_root=/data/zhifan/data/nerf_synthetic/
export CUDA_VISIBLE_DEVICES=7
old_tag=vanilla_adapt_bg_0.2_center_loss_e35k_nodecay_asize
new_tag=vanilla_adapt_bg_0.4_center_loss_e35k_nodecay_asize

set -x

mkdir ckpt_tanks/${new_tag}
for scene in Barn Caterpillar Family Ignatius Truck
do
    old_dir=ckpt_tanks/${old_tag}
    new_dir=ckpt_tanks/${new_tag}
    cp -r ${old_dir}/${scene} ${new_dir}/${scene}
    ${set_cuda} python examples/extract_mesh.py --scene ${scene} --load_path ${new_dir}/${scene} --mesh_level 0.4
done
