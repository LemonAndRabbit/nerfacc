#!/bin/bash

data_root=/data/zhifan/data/nerf_synthetic/
export CUDA_VISIBLE_DEVICES=7
tag=vanilla_random_bg_0.1_center_loss_e35k_nodecay_asize

set -x

# for scene in Wineholder Steamtrain Spaceship Bike Robot Lifestyle
# do
#     dir=ckpt_nsvf/${tag}
#     ${set_cuda} python examples/extract_mesh.py --scene ${scene} --load_path ${dir}/${scene} --mesh_level 0.1
# done

# for scene in  Palace Toad
# do
#     dir=ckpt_nsvf/${tag}
#     ${set_cuda} python examples/extract_mesh.py --scene ${scene} --load_path ${dir}/${scene} --mesh_level 0.4
# done

for scene in Palace
do
    dir=ckpt_nsvf/${tag}
    ${set_cuda} python examples/extract_mesh.py --scene ${scene} --load_path ${dir}/${scene} --mesh_level 0.6
done
