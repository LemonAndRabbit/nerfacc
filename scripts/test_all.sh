#!/bin/bash
set -x
export CUDA_VISIBLE_DEVICES=8
# for epoch in 5000 10000 15000 20000 25000 30000 35000
for epoch in 35000
do
    for scene in chair drums ficus hotdog lego materials mic ship
    do
        dir=mesh_train_epoch${epoch}
        ${set_cuda} python examples/train_ngp_nerf.py --scene ${scene} --load_path ${dir}/${scene}
    done
done