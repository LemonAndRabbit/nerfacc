#!/bin/bash
set -x
export CUDA_VISIBLE_DEVICES=8
for head_layer in 1 2 3
do
    # for head_dim in 16 32 64
    for head_dim in 128
    do
        for scene in chair drums ficus hotdog lego materials mic ship
        do
            dir=mesh_train_head_l${head_layer}_d${head_dim}/${scene}
            mkdir -p ${dir}
            ${set_cuda} python examples/train_ngp_nerf.py --scene ${scene} --save_path ${dir} --train_split train --distortion_loss --color_bkgd_aug random --max_steps 35000 --head_dim ${head_dim} --head_layer ${head_layer} --test_every 35000
            ${set_cuda} python examples/train_ngp_nerf.py --scene ${scene} --load_path ${dir} --export_mesh --mesh_level 0.2 --train_split train
        done
    done
done