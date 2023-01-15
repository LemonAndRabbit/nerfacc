#!/bin/bash
set -x
export CUDA_VISIBLE_DEVICES=0
for scene in chair drums ficus hotdog lego materials mic ship
do
    mkdir -p mesh_0.2_512_hw32_geo15_aug_35k/${scene}
    ${set_cuda} python examples/train_ngp_nerf.py --scene ${scene} --save_path mesh_0.2_512_hw32_geo15_aug_35k/${scene} --train_split train --distortion_loss --color_bkgd_aug random --max_steps 35000 --base_dim 64 --base_layer 2 --head_dim 32 --head_layer 2 --geo_feat_dim 15\
        | tee mesh_0.2_512_hw32_geo15_aug_35k/${scene}/log.txt
    ${set_cuda} python examples/train_ngp_nerf.py --scene ${scene} --load_path mesh_0.2_512_hw32_geo15_aug_35k/${scene} --export_mesh --mesh_level 0.2 --train_split train
done