import math

import numpy as np

import torch
import torch.nn as nn

import torch.nn.functional as F

import torch.utils


def checkpoint(f, *args, **kwargs):
    return f(*args, **kwargs)


def attention(query, key, value, attn_mask=None, need_weights=True, dropout=None):
    # https://pytorchnlp.readthedocs.io/en/latest/_modules/torchnlp/nn/attention.html
    # Needs [batch, heads, seqlen, hid]

    batch_size, heads, query_len, dim = query.size()
    key_len = key.size(2)

    # Scaling by dim due to http://nlp.seas.harvard.edu/2018/04/03/attention.html
    attention_scores = torch.matmul(query, key.transpose(-1, -2).contiguous()) / math.sqrt(dim)
    if attn_mask is not None:
        attn_mask = attn_mask.view(1, 1, *attn_mask.shape[-2:])
        attention_scores = attention_scores + attn_mask  # Mask is additive and contains -Infs

    attention_weights = F.softmax(attention_scores, dim=-1)
    if dropout:
        attention_weights = dropout(attention_weights)
    attention_weights = attention_weights.view(batch_size, heads, query_len, key_len)

    mix = torch.matmul(attention_weights, value)
    return mix, attention_weights


class Overparam(nn.Module):
    def __init__(self, nhid):
        super().__init__()
        self.l1 = nn.Linear(nhid, 2 * nhid)
        # self.l2 = nn.Linear(2 * nhid, 2 * nhid)
        self.inner_act = torch.tanh  # GELU()
        self.nhid = nhid

    def forward(self, x):
        c, f = self.l1(x).split(self.nhid, dim=-1)
        # c, f = self.l2(self.inner_act(self.l1(x))).split(self.nhid, dim=-1)
        return torch.sigmoid(f) * torch.tanh(c)


class Attention(nn.Module):
    def __init__(self, nhid, q=True, k=False, v=False, r=False, heads=1, dropout=None, batch_first=False):
        super().__init__()
        self.qs = nn.Parameter(torch.zeros(size=(1, 1, nhid), dtype=torch.float))
        self.ks = nn.Parameter(torch.zeros(size=(1, 1, nhid), dtype=torch.float))
        self.vs = nn.Parameter(torch.zeros(size=(1, 1, nhid), dtype=torch.float))
        self.qkvs = nn.Parameter(torch.zeros(size=(1, 3, nhid), dtype=torch.float))
        self.heads = heads
        self.nhid = nhid
        assert nhid % self.heads == 0, 'Heads must divide vector evenly'
        self.drop = nn.Dropout(dropout) if dropout else None
        self.gelu = GELU()
        self.q = nn.Linear(nhid, nhid) if q else None
        # self.qln = nn.LayerNorm(nhid, eps=1e-12)
        self.k = nn.Linear(nhid, nhid) if k else None
        self.v = nn.Linear(nhid, nhid) if v else None
        self.r = nn.Linear(2 * nhid, nhid) if r else None
        self.r_gate = nn.Parameter(torch.ones(size=(1, 1, nhid), dtype=torch.float))
        self.vq_store = None
        self.vq = Overparam(nhid)
        # from fastai.text.models import QRNNLayer
        # self.vq = QRNNLayer(input_size=nhid, hidden_size=nhid, save_prev_x=False,
        #                     zoneout=0, window=1, output_gate=False, batch_first=False)
        self.vq_collapsed = False
        self.batch_first = batch_first

    def vq_collapse(self):
        self.vq_store = self.vq
        self.vq = None
        self.vq_collapsed = True

    def vq_uncollapse(self):
        self.vq = self.vq_store
        self.vq_collapsed = False

    def forward(self, query, key, value, attn_mask=None, **kwargs):
        # tanh on the value allows us to flip the polarity of the output, helping use the full range
        # Discovered accidentally when I used QRNN_with_tanh_output(sigmoid(vs))
        # qs, ks, vs = torch.sigmoid(self.qs), torch.sigmoid(self.ks), self.vs
        qs, ks, vs = torch.sigmoid(self.qs), torch.sigmoid(self.ks), torch.sigmoid(self.vs)
        # qs, ks, vs = self.qs, self.ks, self.vs
        # vs = torch.tanh(self.vs)
        if self.vq:
            # vs, _ = self.vq(vs)
            vs = self.vq(vs)
            # qs, ks, vs = [x.reshape((1, 1, -1)) for x in self.vq(torch.sigmoid(self.qkvs))[0, :]]
        elif self.vq_collapsed:
            vs = self.vs
        # qs, ks, vs = self.qs, self.ks, self.vs
        # q = qs * query
        # if self.q: query = self.q(query)
        if self.q:
            query = self.q(query)
            # query = self.qln(query.float())
        if self.k:
            key = self.k(key)
        if self.v:
            value = self.v(value)
        # This essentially scales everything to zero to begin with and then learns from there
        # q, k, v = self.qs * query, self.ks * key, self.vs * value
        q, k, v = qs * query, ks * key, vs * value
        # q, k, v = query, key, vs * value
        # q, k, v = qs * query, ks * key, value
        # k, v = ks * key, vs * value
        # q, k, v = query, key, value
        if self.drop:
            # We won't apply dropout to v as we can let the caller decide if dropout should be applied to the output
            # Applying dropout to q is equivalent to the same mask on k as they're "zipped"
            # q, k, v = self.drop(q), k, v
            q, k, v = self.drop(q), k, self.drop(v)

        original_q = q

        if not self.batch_first:
            q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)

        batch_size, query_len, nhid = q.size()
        assert nhid == self.nhid
        key_len = k.size(1)
        ###
        dim = self.nhid // self.heads
        q = q.view(batch_size, query_len, self.heads, dim).transpose(1, 2)
        k, v = [vec.view(batch_size, key_len, self.heads, dim).transpose(1, 2) for vec in [k, v]]

        mix, focus = attention(q, k, v, dropout=self.drop, attn_mask=attn_mask, **kwargs)
        mix = mix.transpose(1, 2).contiguous().view(batch_size, -1, self.nhid)
        if not self.batch_first:
            mix = mix.transpose(0, 1)

        if self.r:
            # The result should be transformed according to the query
            r = torch.cat([mix, original_q], dim=-1)
            if self.drop:
                r = self.drop(r)
            r = self.gelu(self.r(r))
            mix = torch.sigmoid(self.r_gate) * mix + r
            # BUG: This does _nothing_ as mix isn't set to r ...
            # But ... I got good results with this ... so ...
            # Let's leave it as is for right now ...
            # This does imply that I don't necessarily need complex post mixing ops

        return mix, focus


class PyTorchAttention(nn.Module):
    def __init__(self, nhid, q=True, k=False, v=False, heads=1, dropout=None):
        super().__init__()
        self.mha = nn.MultiheadAttention(nhid, heads, dropout=dropout)

    def forward(self, q, k, v, attn_mask=None):
        return self.mha(q, k, v, attn_mask=attn_mask)


class Block(nn.Module):
    def __init__(self, embed_dim, hidden_dim, heads=1, dropout=None, rnn=False, residual=True, use_attn=True):
        super().__init__()
        self.attn = None
        if use_attn:
            # self.attn = PyTorchAttention(embed_dim, heads=heads, dropout=0)
            self.attn = Attention(embed_dim, heads=heads, r=False, batch_first=True)
        self.ff = Boom(embed_dim, hidden_dim, dropout=dropout, shortcut=True)
        self.lnstart = nn.LayerNorm(embed_dim, eps=1e-12)
        self.lnmid = nn.LayerNorm(embed_dim, eps=1e-12)
        self.lnmem = nn.LayerNorm(embed_dim, eps=1e-12)
        self.lnout = nn.LayerNorm(embed_dim, eps=1e-12)
        self.lnff = nn.LayerNorm(embed_dim, eps=1e-12)
        self.lnxff = nn.LayerNorm(embed_dim, eps=1e-12)
        self.drop = nn.Dropout(0)
        self.gelu = GELU()
        self.residual = residual
        self.rnn = nn.LSTM(input_size=embed_dim, hidden_size=embed_dim, batch_first=True)

    def forward(self, h, attn_mask, mem=None, hidden=None):
        new_mem = None

        h = self.lnstart(h)

        if self.rnn:
            x, new_hidden = self.rnn(h, None if hidden is None else hidden)
            h = h + x if self.residual else x.float()

        focus, new_mem = None, []
        if self.attn is not None:
            z = self.lnmem(h)
            h = self.lnmid(h)

            # if mem is not None:
            #     bigh = torch.cat([mem, mh], dim=0)
            # else:

            q, k, v = h, z, z

            x, focus = checkpoint(self.attn, q, k, v, attn_mask)
            # x, focus = tcheckpoint(self.attn, q, k, bigh, attn_mask)
            # x = self.drop(x)
            h = x + h

        h, x = self.lnff(h), self.lnxff(h)
        x = checkpoint(self.ff, x)
        # x = tcheckpoint(self.ff, h)
        # x = self.drop(x)
        h = x + h

        return h, None, new_hidden, focus


class SHARNN(nn.Module):
    def __init__(self, ntoken, ninp, nhid, nlayers, dropouth=0.5):
        super().__init__()
        embed_dim = ninp
        hidden_dim = nhid
        self.ninp, self.nhid = ninp, nhid
        self.nlayers = nlayers
        num_embeddings = ntoken
        self.num_heads = 1  # 4

        self.blocks = nn.ModuleList()
        for idx in range(nlayers):
            self.blocks.append(Block(embed_dim, hidden_dim, self.num_heads, dropout=dropouth,
                                     rnn=True, residual=False, use_attn=True))

        self.encoder = nn.Linear(num_embeddings, embed_dim)
        self.decoder = nn.Linear(embed_dim, num_embeddings)

        # self.apply(self.init_weights)

    def eval(self):
        super().eval()
        # for block in self.blocks:
        #     block.attn.vq_collapse()

    def train(self, mode=True):
        super().train(mode)
        # for block in self.blocks:
        #     block.attn.vq_uncollapse()

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding, nn.LayerNorm)):
            module.weight.data.normal_(mean=0.0, std=0.1 / np.sqrt(self.ninp))

        if isinstance(module, (nn.Linear, nn.LayerNorm)) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, x):
        """ Input has shape [seq length, batch] """
        h = self.encoder(x.permute(0, 2, 1))

        # attn_mask = torch.full((x.shape[-1], x.shape[-1]), -float('Inf'), device=h.device, dtype=h.dtype)
        # attn_mask = torch.triu(attn_mask, diagonal=1)

        for idx, block in enumerate(self.blocks):
            # p = torch.sigmoid(self.position_gates[idx]) * pe
            h, m, nh, f = block(h, attn_mask=None, mem=None, hidden=None)
            # focus.append(f)

        h = self.decoder(h)

        return h.permute(0, 2, 1)


class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """
    def forward(self, x):
        # return torch.nn.functional.gelu(x.float())
        # The first approximation has more operations than the second
        # See https://arxiv.org/abs/1606.08415
        # return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        return x * torch.sigmoid(1.702 * x)

# @torch.jit.script
# def GELU(x):
#   return x * torch.sigmoid(1.702 * x)


class Boom(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1, shortcut=False):
        super(Boom, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout) if dropout else None
        if not shortcut:
            self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.shortcut = shortcut
        # self.act = nn.ReLU()
        self.act = GELU()
        # self.act = nn.Tanh()

    def forward(self, input):
        x = self.act(self.linear1(input))
        if self.dropout:
            x = self.dropout(x)
        if self.shortcut:
            # Trim the end off if the size is different
            ninp = input.shape[-1]
            x = torch.narrow(x, -1, 0, x.shape[-1] // ninp * ninp)
            # Divide the hidden size evenly into chunks
            x = x.view(*x.shape[:-1], x.shape[-1] // ninp, ninp)
            # Collapse the chunks through summation
            # h = h + self.drop(x).sum(dim=-2)
            z = x.sum(dim=-2)
        else:
            z = self.linear2(x)

        return z
