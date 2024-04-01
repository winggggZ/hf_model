#!/bin/bash

for batch in 64 80
do
  deepspeed llama2_layer_deepspeed_dp.py \
              --deepspeed --deepspeed_config Llama2_config/zero_0_${batch}.json \
              --iter 100 \
              --config Llama2_config/Llama2_7B_2layers.json \
              --logdir log \
              >> time_log/deepspeed_Llama_7B_8GPU_2layers.log
done

for batch in 48 64 128
do
  deepspeed llama2_layer_deepspeed_dp.py \
              --deepspeed --deepspeed_config Llama2_config/zero_0_${batch}.json \
              --iter 100 \
              --config Llama2_config/Llama2_7B_4layers.json \
              --logdir log \
              >> time_log/deepspeed_Llama_7B_8GPU_4layers.log
done

for batch in 16 64 96 128
do
  deepspeed llama2_layer_deepspeed_dp.py \
              --deepspeed --deepspeed_config Llama2_config/zero_0_${batch}.json \
              --iter 100 \
              --config Llama2_config/Llama2_34B_2layers.json \
              --logdir log \
              >> time_log/deepspeed_Llama_34B_8GPU_2layers.log
done