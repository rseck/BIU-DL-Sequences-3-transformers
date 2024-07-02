from tests import DEBUG
from typing import Optional
from torch import nn
import torch
import torch.nn.functional as F
import math


def create_kqv_matrix(input_vector_dim, n_heads=1):
    linear = nn.Linear(input_vector_dim, int((3 * (input_vector_dim / n_heads))))
    if DEBUG:
        nn.init.constant_(linear.weight, 1.0)
        nn.init.constant_(linear.bias, 0.0)
    return linear


def kqv(x, linear):
    B, N, D = x.size()
    k, q, v = torch.split(linear(x), 3, dim=2)
    return k, q, v


def attention_scores(a, b):

    B1, N1, D1 = a.size()
    B2, N2, D2 = b.size()
    assert B1 == B2
    assert D1 == D2

    return b @ a.transpose(1, 2) / math.sqrt(D1)


def create_causal_mask(embed_dim, n_heads, max_context_len):
    return torch.tril(torch.ones((1, max_context_len, max_context_len)))


def self_attention(v, A, mask=None):
    B1, N1, D1 = v.size()
    B2, N2, D2 = A.size()
    assert B1 == B2
    assert N1 == N2
    assert N1 == D2
    if mask is not None:
        M = mask[0, :N2, :N2]
        A = A.masked_fill(M == 0, float("-inf"))
    # softmax over rows of attention matrix, q_i is constant while k_j varies
    return torch.nn.functional.softmax(A, dim=2) @ v


def self_attention_layer(x, kqv_matrix, attention_mask):
    k, q, v = kqv(x, kqv_matrix)
    att = attention_scores(k, q)
    sa = self_attention(v, att, attention_mask)
    return sa


def multi_head_attention_layer(x, kqv_matrices, mask):
    raise Exception("Not implemented.")
    B, N, D = x.size()
    # TODO implement multi-head attention.
    # This is most easily done using calls to self_attention_layer, each with a different
    # entry in kqv_matrices, and combining the results.
    #
    # There is also a tricker (but more efficient) version of multi-head attention, where we do all the computation
    # using a single multiplication with a single kqv_matrix (or a single kqv_tensor) and re-arranging the results afterwards.
    # If you want a challenge, you can try and implement this. You may need to change additional places in the code accordingly.
    assert sa.size() == x.size()
    return sa


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, n_heads, max_context_len):
        super().__init__()
        assert embed_dim % n_heads == 0
        # the linear layers used for k, q, v computations:
        # each linear is for a different head, but for all of k, q and v for this head.
        self.kqv_matrices = nn.ModuleList([create_kqv_matrix(embed_dim, n_heads) for i in range(n_heads)])
        # For use in the causal part.  "register_buffer" is used to store a tensor which is fixed but is not a parameter of the model.
        # You can then access it with: self.mask
        mask = create_causal_mask(embed_dim, n_heads, max_context_len)
        self.register_buffer("mask", mask)
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        sa = multi_head_attention_layer(x, self.kqv_matrices, self.mask)
        sa = self.proj(sa)
        return sa
