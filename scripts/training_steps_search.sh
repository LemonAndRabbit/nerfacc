#!/bin/bash
set -x
export CUDA_VISIBLE_DEVICES=8
for epoch in 5000 10000 15000 20000 25000 30000 35000
do
    for scene in chair drums ficus hotdog lego materials mic ship
    do
        dir=mesh_train_epoch${epoch}
        mkdir -p ${dir}/${scene}
        ${set_cuda} python examples/train_ngp_nerf.py --scene ${scene} --save_path ${dir}/${scene} --train_split train --distortion_loss --color_bkgd_aug random --max_steps ${epoch} --test_every ${epoch}
        ${set_cuda} python examples/train_ngp_nerf.py --scene ${scene} --load_path ${dir}/${scene} --export_mesh --mesh_level 0.2
    done
done