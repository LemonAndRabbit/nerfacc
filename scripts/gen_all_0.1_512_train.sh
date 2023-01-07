#!/bin/bash
set -x
export CUDA_VISIBLE_DEVICES=7
for scene in chair drums ficus hotdog lego materials mic ship; do
    mkdir -p mesh_0.1_512_train/${scene}
    ${set_cuda} python examples/train_ngp_nerf.py --scene ${scene} --save_path mesh_0.1_512_train/${scene} --train_split train --distortion_loss
    ${set_cuda} python examples/train_ngp_nerf.py --scene ${scene} --load_path mesh_0.1_512_train/${scene} --export_mesh --mesh_level 0.1 --train_split train
done
