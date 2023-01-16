#!/bin/bash
set -x
export CUDA_VISIBLE_DEVICES=9
for epoch in 35000
do
    for scene in chair drums ficus hotdog lego materials mic ship
    do
        dir=time_test_epoch${epoch}
        mkdir -p ${dir}/${scene}
        ${set_cuda} python examples/train_ngp_nerf.py --scene ${scene} --save_path ${dir}/${scene} --train_split train --distortion_loss --color_bkgd_aug random --max_steps ${epoch} --test_every 35001
    done
done