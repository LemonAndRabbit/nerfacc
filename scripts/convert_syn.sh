#!/bin/bash
set -x
prefix=../inherite_super_40_12_16
out_path=../inherite_super_40_12_16

for scene in chair drums ficus hotdog lego materials mic ship
do
    mkdir -p ${out_path}/${scene}
    ${set_cuda} python examples/model_converter.py --scene ${scene} --load_path ${prefix}/${scene} --save_path ${out_path}/${scene} --n_level 12 --hash_size 16
done
