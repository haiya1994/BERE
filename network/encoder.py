import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init

from network import utils as utils

class MultiAttn(nn.Module):
    def __init__(self, in_dim, head_num=10):
        super(MultiAttn, self).__init__()

        self.head_dim = in_dim // head_num
        self.head_num = head_num

        # scaled dot product attention
        self.scale = self.head_dim ** -0.5

        self.w_qs = nn.Linear(in_dim, head_num * self.head_dim, bias=True)
        self.w_ks = nn.Linear(in_dim, head_num * self.head_dim, bias=True)
        self.w_vs = nn.Linear(in_dim, head_num * self.head_dim, bias=True)

        self.w_os = nn.Linear(head_num * self.head_dim, in_dim, bias=True)

        self.gamma = nn.Parameter(torch.FloatTensor([0]))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, attn_mask, non_pad_mask):
        B, L, H = x.size()
        head_num = self.head_num
        head_dim = self.head_dim

        q = self.w_qs(x).view(B * head_num, L, head_dim)
        k = self.w_ks(x).view(B * head_num, L, head_dim)
        v = self.w_vs(x).view(B * head_num, L, head_dim)

        attn_mask = attn_mask.repeat(head_num, 1, 1)

        attn = torch.bmm(q, k.transpose(1, 2))  # B*head_num, L, L
        attn = self.scale * attn
        attn = attn.masked_fill_(attn_mask, -np.inf)
        attn = self.softmax(attn)

        out = torch.bmm(attn, v)  # B*head_num, L, head_dim

        out = out.view(B, L, head_dim * head_num)

        out = self.w_os(out)

        out = non_pad_mask * out

        out = self.gamma * out + x

        return out, attn


class PackedGRU(nn.Module):
    def __init__(self, in_dim, hid_dim, bidirectional=True):
        super(PackedGRU, self).__init__()

        self.gru = nn.GRU(in_dim, hid_dim, batch_first=True, bidirectional=bidirectional)

    def forward(self, x, length):
        packed = torch.nn.utils.rnn.pack_padded_sequence(x, length, batch_first=True)
        out, _ = self.gru(packed)
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        return out


class LeafRNN(nn.Module):
    def __init__(self, in_dim, hid_dim, bidirectional=True):
        super(LeafRNN, self).__init__()
        self.bidirectional = bidirectional

        self.leaf_rnn = nn.GRU(in_dim, hid_dim, batch_first=True)

        if self.bidirectional:
            self.leaf_rnn_bw = nn.GRU(in_dim, hid_dim, batch_first=True)

    def forward(self, x, non_pad_mask, length=None):
        out, _ = self.leaf_rnn(x)
        out = non_pad_mask * out

        if self.bidirectional:
            in_bw = utils.reverse_padded_sequence(x, length, batch_first=True)
            out_bw, _ = self.leaf_rnn_bw(in_bw)
            out_bw = non_pad_mask * out_bw
            out_bw = utils.reverse_padded_sequence(out_bw, length, batch_first=True)
            out = torch.cat([out, out_bw], -1)

        return out


class BinaryTreeGRULayer(nn.Module):
    def __init__(self, hidden_dim):
        super(BinaryTreeGRULayer, self).__init__()

        self.fc1 = nn.Linear(in_features=2 * hidden_dim, out_features=3 * hidden_dim)
        self.fc2 = nn.Linear(in_features=2 * hidden_dim, out_features=hidden_dim)

    def forward(self, hl, hr):
        """
        Args:
            hl: (batch_size, max_length, hidden_dim).
            hr: (batch_size, max_length, hidden_dim).
        Returns:
            h: (batch_size, max_length, hidden_dim).
        """

        hlr_cat1 = torch.cat([hl, hr], dim=-1)
        treegru_vector = self.fc1(hlr_cat1)
        i, f, r = treegru_vector.chunk(chunks=3, dim=-1)

        hlr_cat2 = torch.cat([hl * r.sigmoid(), hr * r.sigmoid()], dim=-1)

        h_hat = self.fc2(hlr_cat2)

        h = (hl + hr) * f.sigmoid() + h_hat.tanh() * i.sigmoid()

        return h


class GumbelTreeGRU(nn.Module):
    def __init__(self, hidden_dim):
        super(GumbelTreeGRU, self).__init__()
        self.hidden_dim = hidden_dim

        self.gumbel_temperature = nn.Parameter(torch.FloatTensor([1]))

        self.treegru_layer = BinaryTreeGRULayer(hidden_dim)

        self.comp_query = nn.Parameter(torch.FloatTensor(hidden_dim))
        init.normal_(self.comp_query.data, mean=0, std=0.01)

        self.query_layer = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 10, bias=True), nn.Tanh(),
                                         nn.Linear(hidden_dim // 10, 1, bias=True))

    @staticmethod
    def update_state(old_h, new_h, done_mask):
        done_mask = done_mask.float().unsqueeze(1).unsqueeze(2)
        h = done_mask * new_h + (1 - done_mask) * old_h[:, :-1, :]
        return h

    def select_composition(self, old_h, new_h, mask):
        old_h_left, old_h_right = old_h[:, :-1, :], old_h[:, 1:, :]

        comp_weights = self.query_layer(new_h).squeeze(2)


        if self.training:
            select_mask = utils.st_gumbel_softmax(
                logits=comp_weights, temperature=self.gumbel_temperature,
                mask=mask)
        else:
            select_mask = utils.greedy_select(logits=comp_weights, mask=mask).float()

        select_mask_cumsum = select_mask.cumsum(1)
        left_mask = 1 - select_mask_cumsum
        right_mask = select_mask_cumsum - select_mask

        new_h = (select_mask.unsqueeze(2) * new_h
                 + left_mask.unsqueeze(2) * old_h_left
                 + right_mask.unsqueeze(2) * old_h_right)

        return new_h, select_mask

    def forward(self, input, length):
        max_depth = input.size(1)
        length_mask = utils.sequence_mask(length=length, max_length=max_depth)
        select_masks = []

        h = input

        for i in range(max_depth - 1):
            hl = h[:, :-1, :]
            hr = h[:, 1:, :]
            new_h = self.treegru_layer(hl, hr)
            if i < max_depth - 2:
                # We don't need to greedily select the composition in the
                # last iteration, since it has only one option left.
                new_h, select_mask = self.select_composition(
                    old_h=h, new_h=new_h,
                    mask=length_mask[:, i + 1:])

                select_masks.append(select_mask)

            done_mask = length_mask[:, i + 1]

            h = self.update_state(old_h=h, new_h=new_h,
                                  done_mask=done_mask)

        out = h.squeeze(1)

        return out, select_masks
