import torch
from torch import Tensor
from attention import CausalSelfAttention, self_attention_layer
from pathlib import Path

def heat_map_attention(x: Tensor, attention_layer: CausalSelfAttention):
    for kqv_matrix in attention_layer.kqv_matrices:
        heat_map = self_attention_layer(x, kqv_matrix, attention_layer.mask)


def main(model_path: str):
    # load model with cpu and create heat map from the attention heads
    model = torch.load(model_path)
    for layer in model.layers:
        if isinstance(layer, CausalSelfAttention):
            heat_map_attention(layer.x, layer)


if __name__ == "__main__":
    path = Path() / "results for yedidia" / "loss_0.25_llm_model_25000_s_128_b_64_dp_data_nl_6_nh_6_es_192_mhs_768_lr_0.0005_gc_1.0_wd_false_nbtt_25000_us_False.pth"
    main(path)
