import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertConfig
from utils.use import *
import time

device = torch.device('cuda')
par = argparse.ArgumentParser(description='training config')
par.add_argument('--batch', type=int, required=True, help='train batch size')
par.add_argument('--iter', type=int, required=True, help='train iter size')
par.add_argument('--config', type=str, required=True, help='train config')
par.add_argument('--logdir', type=str, required=True, help='train log')
args = par.parse_args()

# config
with open(args.config, 'r') as f:
    config = json.load(f)
config = BertConfig(**config)

# data  batch
inputs = torch.rand(args.batch, config.max_position_embeddings, config.hidden_size).to(device)
label = torch.rand_like(inputs).to(device)
dataset = test_dataset(inputs, label)
training_dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=True)
# model
model = Bert_already_emb(config=config).to(device)
loss_function = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.1)
# pytorch DP
model = nn.DataParallel(model)
# train
total_time, be_time, bs_time, fs_time, fe_time = 0, 0, 0, 0, 0
"""with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=0, warmup=5, active=3, repeat=0),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(dir_name=args.logdir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
        with_modules=True) as prof:"""
for step in range(args.iter):
    for i, (input_emb, label) in enumerate(training_dataloader):
        torch.cuda.synchronize()

        fs_time = time.time()

        out_label = model(input_emb)
        torch.cuda.synchronize()
        # fe_time = time.time()

        loss = loss_function(out_label, label)
        torch.cuda.synchronize()

        # bs_time = time.time()
        loss.backward()
        torch.cuda.synchronize()

        optimizer.step()
        torch.cuda.synchronize()

        optimizer.zero_grad()
        torch.cuda.synchronize()

        be_time = time.time()

        # prof.step()
    if step >= args.iter / 2:
        total_time = total_time + be_time - fs_time

print(f'valid time={be_time - fs_time}')
print(f'Average time={total_time / (args.iter / 2)}')
print(f'batch_size: {args.batch} bert done')
