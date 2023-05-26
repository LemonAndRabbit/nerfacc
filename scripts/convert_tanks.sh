#!/bin/bash
set -x
prefix=../tanks_ckpts
out_path=../tanks_ckpts

for scene in Barn Caterpillar Family Ignatius Truck
do
    mkdir -p ${out_path}/${scene}
    ${set_cuda} python examples/model_converter.py --scene ${scene} --load_path ${prefix}/${scene} --save_path ${out_path}/${scene} --n_level 12 --hash_size 18
done
