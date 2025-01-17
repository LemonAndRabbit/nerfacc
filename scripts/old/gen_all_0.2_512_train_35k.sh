#!/bin/bash
set -x
export CUDA_VISIBLE_DEVICES=4
for scene in chair drums ficus hotdog lego materials mic ship
do
    mkdir -p mesh_0.2_512_train_35k/${scene}
    ${set_cuda} python examples/train_ngp_nerf.py --scene ${scene} --save_path mesh_0.2_512_train_35k/${scene} --train_split train --distortion_loss --max_steps 35000
    ${set_cuda} python examples/train_ngp_nerf.py --scene ${scene} --load_path mesh_0.2_512_train_35k/${scene} --export_mesh --mesh_level 0.2 --train_split train
done