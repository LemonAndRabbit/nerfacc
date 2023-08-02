#!/bin/bash
set -x
data_root=/data/zhifan/data/nerf_synthetic/
export CUDA_VISIBLE_DEVICES=7
for epoch in 35000
do
    # for scene in chair drums ficus hotdog lego materials mic ship
    for scene in chair
    do
        dir=mesh_train_epoch${epoch}
        mkdir -p ${dir}/${scene}
        ${set_cuda} python examples/train_ngp_nerf_occ.py --scene ${scene} --data_root ${data_root} --ckpt_path ${dir}/${scene}
        # ${set_cuda} python examples/train_ngp_nerf.py --scene ${scene} --load_path ${dir}/${scene} --export_mesh --mesh_level 0.2
    done
done