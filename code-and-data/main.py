from __future__ import annotations

import torch
from torch import optim

import data
import lm
from transformer import TransformerLM
import matplotlib.pyplot as plt

if __name__ == "__main__":
    seq_len = 128
    batch_size = 64
    data_path = "data/"
    n_layers = 6
    n_heads = 6
    embed_size = 192
    mlp_hidden_size = embed_size * 4

    learning_rate = 5e-4
    gradient_clipping = 1.0
    weight_decay = 0.01

    num_batches_to_train = 50000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer, tokenized_data = data.load_data(data_path)
    # NOTE: are data items are longer by one than the sequence length,
    # They will be shortened by 1 when converted to training examples.
    data_iter = iter(data.RandomOrderDataIterator(tokenized_data, seq_len + 1))

    model = TransformerLM(
        n_layers,
        n_heads,
        embed_size,
        seq_len,
        tokenizer.vocab_size(),
        mlp_hidden_size,
        with_residuals=True,
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=weight_decay
    )

    model.train()

    num_batches = 0
    losses = []
    while num_batches < num_batches_to_train:
        for batch in data.batch_items(data_iter, batch_size):
            if num_batches >= num_batches_to_train:
                break
            num_batches = num_batches + 1

            batch_x, batch_y = lm.batch_to_labeled_samples(batch)

            logits = model(batch_x.to(device))

            loss = lm.compute_loss(logits, batch_y.to(device))

            # parameters update
            model.zero_grad()
            losses.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            optimizer.step()

            num_batches += 1
            if num_batches % 10 == 0:
                print(f"Seen {num_batches} batches. last loss is: {loss.item()}")
                if num_batches % 100 == 0:
                    for _ in range(1):
                        model.eval()
                        sampled = tokenizer.detokenize(
                            model.sample_continuation(tokenizer.tokenize("Hello"), 500)
                        )
                        model.train()
                        print(f"Model sample: '''{sampled}'''")
                        if num_batches % 10000 == 0:
                            torch.save(model.state_dict(), f"llm_model_{num_batches}.pth")
                    print("")
    plt.plot(losses)  # noqa
    plt.savefig(f"llm_losses.png")
