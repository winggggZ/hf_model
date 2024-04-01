from deepspeed.moe.layer import MoE
import torch
import torch.nn as nn

device = torch.device('cuda')
class moe_layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.x = nn.Linear(20, 20)
        self.moe = MoE(hidden_size=20, expert=self.x,
                       num_experts=5)

    def forward(self, inputs):
        out, _, _ = self.moe(inputs)
        return out


model = moe_layer().to(device)
print(model)
x = torch.rand(16, 20).to(device)
model(x)
