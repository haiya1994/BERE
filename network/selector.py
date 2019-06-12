import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init


class BagAttention(nn.Module):
    def __init__(self, in_dim):
        super(BagAttention, self).__init__()
        self.scale = in_dim ** -0.5

        self.attn_w = nn.Parameter(torch.FloatTensor(in_dim))

        self.reset_parameters()

    def reset_parameters(self):
        init.normal_(self.attn_w.data, mean=0, std=0.01)

    def forward(self, x, scope):
        attn = (self.attn_w * x).sum(-1)
        attn = self.scale * attn  # B

        bag_logits = []
        bag_attns = []
        start_offset = 0
        for i in range(len(scope)):
            end_offset = scope[i]
            bag_x = x[start_offset:end_offset]  # n*H

            bag_attn = F.softmax(attn[start_offset:end_offset], -1)  # n

            bag_attns.append(bag_attn)

            bag_logits.append(torch.matmul(bag_attn, bag_x))  # (n') x (n', hidden_size) = (hidden_size)

            start_offset = end_offset

        bag_logits = torch.stack(bag_logits)

        return bag_logits, bag_attns
