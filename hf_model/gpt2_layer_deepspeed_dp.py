import json
import torch
import torch.optim as optim
from transformers import GPT2Config
from utils.use import *
import time


# cmd_args
cmd_args = parse_args()


# config
with open(cmd_args.config, 'r') as f:
    config = json.load(f)
config = GPT2Config(**config)

with open(cmd_args.deepspeed_config, 'r') as f:
    zero_config = json.load(f)
zero_config = GPT2Config(**zero_config)

# data  batch
data = torch.rand(zero_config.train_batch_size * cmd_args.iter, config.max_position_embeddings, config.hidden_size)
label = torch.ones_like(data)
dataset = test_dataset(data, label)


# model
model = gpt2_already_emb(config=config)
print(model)
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
total_time, f_time, b_time = 0, 0, 0
for i, (x, y) in enumerate(training_dataloader):
    input_emb = x.to(engine.device)
    label = y.to(engine.device)
    fs_time = time.time()
    out_label = engine(input_emb)
    f_time = time.time() - fs_time
    loss = loss_function(out_label, label)
    optimizer.zero_grad()
    bs_time = time.time()
    engine.backward(loss)
    b_time = time.time() - bs_time
    engine.step()
    if i >= cmd_args.iter/2:
        total_time = total_time + f_time + b_time
print(f'valid time={f_time + b_time}')
print(f'Average time={total_time / (cmd_args.iter/2)}')
print(f'batch_size: {zero_config.train_batch_size} gpt done')


