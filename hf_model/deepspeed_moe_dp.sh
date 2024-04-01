#!/bin/bash

for batch in 8 16 32 128
do
    deepspeed --include localhost:4,5,6,7 \
              moe_layer_deepspeed_dp.py \
              --deepspeed --deepspeed_config Moe_config/zero_0_${batch}.json \
              --iter 100 \
              --config Moe_config/moe_7.1B_2layers.json \
              --logdir log \
              >> time_log/deepspeed_Moe_7.1B_4GPU_2layers.log
done

for batch in 8 16 32 48 64
do
    deepspeed --include localhost:4,5,6,7 \
              moe_layer_deepspeed_dp.py \
              --deepspeed --deepspeed_config Moe_config/zero_0_${batch}.json \
              --iter 100 \
              --config Moe_config/moe_7.1B_4layers.json \
              --logdir log \
              >> time_log/deepspeed_Moe_7.1B_4GPU_4layers.log
done

for batch in 16 32 48 112 128
do
    deepspeed moe_layer_deepspeed_dp.py \
              --deepspeed --deepspeed_config Moe_config/zero_0_${batch}.json \
              --iter 100 \
              --config Moe_config/moe_7.1B_2layers.json \
              --logdir log \
              >> time_log/deepspeed_Moe_7.1B_8GPU_2layers.log
done

for batch in 16 32 48 64 80 96 112 128
do
    deepspeed moe_layer_deepspeed_dp.py \
              --deepspeed --deepspeed_config Moe_config/zero_0_${batch}.json \
              --iter 100 \
              --config Moe_config/moe_2.4B_2layers.json \
              --logdir log \
              >> time_log/deepspeed_Moe_2.4B_8GPU_2layers.log
done

for batch in 16 32 48 64 80 96 112 128
do
    deepspeed moe_layer_deepspeed_dp.py \
              --deepspeed --deepspeed_config Moe_config/zero_0_${batch}.json \
              --iter 100 \
              --config Moe_config/moe_10B_2layers.json \
              --logdir log \
              >> time_log/deepspeed_Moe_10B_8GPU_2layers.log
done


for batch in 16 32 48 64
do
    deepspeed moe_layer_deepspeed_dp.py \
              --deepspeed --deepspeed_config Moe_config/zero_0_${batch}.json \
              --iter 100 \
              --config Moe_config/moe_7.1B_6layers.json \
              --logdir log \
              >> time_log/deepspeed_Moe_7.1B_8GPU_6layers.log
done








