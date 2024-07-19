from torch import Tensor
from attention import CausalSelfAttention, self_attention_layer


def heat_map_attention(x: Tensor, attention_layer: CausalSelfAttention):
    for kqv_matrix in attention_layer.kqv_matrices:
        heat_map = self_attention_layer(x, kqv_matrix, attention_layer.mask)


def main(model_path: str):
    torch.load

