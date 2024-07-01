import torch
from torch import nn

import attention

DEBUG = False


def test_attention_scores():
    # fill in values for the a, b and expected_output tensor.
    a = torch.tensor([])  # a three-dim tensor
    b = torch.tensor([])  # a three-dim tensor
    expected_output = torch.tensor([])  # a three-dim tensor

    A = attention.attention_scores(a, b)

    # Note that we use "allclose" and not ==, so we are less sensitive to float inaccuracies.
    assert torch.allclose(A, expected_output)


def test_kqv():
    n_heads = 2
    B = batch_size = 3
    N = sequence_length = 4
    D = embeddings_dimension = n_heads * 3
    x = torch.ones(B, N, D)
    kqv_matrices = nn.ModuleList([attention.create_kqv_matrix(D, n_heads) for i in range(n_heads)])
    for kqv_matrix in kqv_matrices:
        k, q, v = attention.kqv(x, kqv_matrix)
        assert k.shape == (B, N, int(D / n_heads))
        assert q.shape == (B, N, int(D / n_heads))
        assert v.shape == (B, N, int(D / n_heads))
        if DEBUG:
            expected_output = torch.ones(B, N, int(D / n_heads)) * D
            assert torch.equal(expected_output, k)
            assert torch.equal(expected_output, q)
            assert torch.equal(expected_output, v)


if __name__ == '__main__':
    test_kqv()
