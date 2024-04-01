#!/bin/bash

deepspeed --include localhost:4,5,6,7 \
          gpt2_layer_deepspeed_dp.py \
              --deepspeed --deepspeed_config GPT_config/zero_0.json \
              --iter 10 \
              --config GPT_config/gpt2_15B.json \
              --logdir log