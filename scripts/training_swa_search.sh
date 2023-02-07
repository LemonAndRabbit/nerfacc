#!/bin/bash
set -x
export CUDA_VISIBLE_DEVICES=8
for epoch in 35000
do
    for scene in chair drums ficus hotdog lego materials mic ship
    do
        dir=mesh_swa_train_epoch${epoch}
        mkdir -p ${dir}/${scene}
        ${set_cuda} python examples/train_ngp_nerf.py --scene ${scene} --save_path ${dir}/${scene} --train_split train --distortion_loss --color_bkgd_aug random --max_steps ${epoch} --swa --test_every 35000
        ${set_cuda} python examples/train_ngp_nerf.py --scene ${scene} --load_path ${dir}/${scene} --export_mesh --mesh_level 0.2
    done
done