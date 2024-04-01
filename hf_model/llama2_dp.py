import torch
from datasets import load_dataset
from transformers import BertTokenizer, LlamaConfig
from utils.use import *
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import json


device = torch.device('cuda')
# path
log_dir = 'log2/llama2_dp'

# data_set
tokenizer = BertTokenizer(vocab_file='bert-base-uncased/vocab.txt')
raw_datasets = load_dataset(path='imdb_dataset')
text = raw_datasets['train']['text'][0:640]
token = tokenizer(text, padding='max_length', truncation=True, return_tensors='pt', max_length=128)
label = torch.ones(640, 10,dtype=torch.long)
data = my_dataset(token, label)
training_dataloader = DataLoader(data, batch_size=64, shuffle=True)

# model
with open('Llama2_config/Llama2_7B.json', 'r') as f:
    config = json.load(f)
config = LlamaConfig(**config)
model = Llama2(config=config).to(device)
loss_function = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.1)

# pytorch DP
model = nn.DataParallel(model)


# train
with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=0, warmup=5, active=3, repeat=0),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(dir_name=log_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
        with_modules=True) as prof:
    for i, (x, y, z, m) in enumerate(training_dataloader):
        input_ids = x.to(device)
        attention_mask = y.to(device)
        token_type_ids = z.to(device)
        label = m.to(device)
        out_label = model(input_ids, attention_mask, token_type_ids)
        loss = loss_function(out_label, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        prof.step()
