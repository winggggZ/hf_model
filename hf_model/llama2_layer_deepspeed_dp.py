import json
import torch
import torch.optim as optim
from transformers import LlamaConfig
from utils.use import *
import time

# cmd_args
cmd_args = parse_args()

# config
with open(cmd_args.config, 'r') as f:
    config = json.load(f)
config = LlamaConfig(**config)

with open(cmd_args.deepspeed_config, 'r') as f:
    zero_config = json.load(f)
zero_config = LlamaConfig(**zero_config)

# data  batch
data = torch.rand(zero_config.train_batch_size, config.seq_len, config.hidden_size)
label = torch.ones_like(data)
dataset = test_dataset(data, label)

# model
model = Llama2_already_emb(config=config)
optimizer = optim.SGD(model.parameters(), lr=0.1)
engine, optimizer, training_dataloader, _ = deepspeed.initialize(
    args=cmd_args,
    model=model,
    model_parameters=model.parameters(),
    optimizer=optimizer,
    training_data=dataset
)
loss_function = nn.CrossEntropyLoss().to(engine.device)

# train
total_time, fs_time, fe_time, be_time, bs_time = 0, 0, 0, 0, 0
"""with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=0, warmup=5, active=3, repeat=0),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(dir_name=cmd_args.logdir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
        with_modules=True) as prof:"""
for epoch in range(cmd_args.iter):
    for i, (x, y) in enumerate(training_dataloader):
        input_emb = x.to(engine.device)
        label = y.to(engine.device)

        torch.cuda.synchronize()
        fs_time = time.time()

        out_label = engine(input_emb)
        torch.cuda.synchronize()
        # fe_time = time.time()

        loss = loss_function(out_label, label)
        torch.cuda.synchronize()

        # bs_time = time.time()
        engine.backward(loss)
        torch.cuda.synchronize()

        engine.step()
        torch.cuda.synchronize()
        optimizer.zero_grad()
        torch.cuda.synchronize()

        be_time = time.time()

        # prof.step()

    if epoch >= cmd_args.iter / 2:
        total_time = total_time + be_time - fs_time

print(f'valid time={be_time - fs_time}')
print(f'Average time={total_time / (cmd_args.iter / 2)}')
print(f'batch_size: {zero_config.train_batch_size} llama done')
