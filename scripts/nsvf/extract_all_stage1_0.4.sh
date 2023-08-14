#!/bin/bash

data_root=/data/zhifan/data/nerf_synthetic/
export CUDA_VISIBLE_DEVICES=7
old_tag=vanilla_random_bg_0.1_center_loss_e35k_nodecay_asize
new_tag=vanilla_random_bg_0.4_center_loss_e35k_nodecay_asize

set -x

mkdir ckpt_nsvf/${new_tag}
for scene in Wineholder Steamtrain Spaceship Bike Robot Lifestyle
do
    old_dir=ckpt_nsvf/${old_tag}
    new_dir=ckpt_nsvf/${new_tag}
    cp -r ${old_dir}/${scene} ${new_dir}/${scene}
    ${set_cuda} python examples/extract_mesh.py --scene ${scene} --load_path ${new_dir}/${scene} --mesh_level 0.4
done

# for scene in Toad Palace
# do
#     dir=ckpt_nsvf/${tag}
#     ${set_cuda} python examples/extract_mesh.py --scene ${scene} --load_path ${dir}/${scene} --mesh_level 0.4
# done