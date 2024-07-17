from __future__ import annotations

import json
import os.path

import torch
from torch import optim
import numpy as np

import data
import lm
from transformer import TransformerLM
import matplotlib.pyplot as plt


def get_file_name(seq_len, batch_size, data_path, n_layers, n_heads, embed_size, mlp_hidden_size, learning_rate,
                  gradient_clipping, weight_decay, num_batches_to_train, use_scheduler):
    s = seq_len
    b = batch_size
    dp = data_path
    nl = n_layers
    nh = n_heads
    es = embed_size
    mhs = mlp_hidden_size

    lr = learning_rate
    gc = gradient_clipping
    wd = weight_decay

    nbtt = num_batches_to_train
    us = use_scheduler

    return (
        f"without_proper_init_with_dropout_s_{s}_b_{b}_dp_{dp}_nl_{nl}_nh_{nh}_es_{es}_mhs_{mhs}_lr_{lr}_gc_{gc}_wd_{wd}_nbtt_{nbtt}_us_{us}".
        replace('/', ''))


def save_plot_of_loss_on_train_and_test(train_loss, test_loss, results_path, run_file_name):
    A = train_loss
    B = test_loss

    # X-axis for A
    x_A = np.arange(len(A))

    # X-axis for B, taking every 100th value
    x_B = np.arange(0, len(A), 100)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(x_A, A, label='train loss')
    plt.scatter(x_B, B, color='red', label='test loss', zorder=5)

    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Plot of train and test losses')
    plt.legend()
    plt.savefig(os.path.join(results_path, f"llm_losses_on_train_and_test_{run_file_name}.png"))


def main():
    seq_len = 128
    batch_size = 128
    data_path = "data/"
    train_path = "data/train/"
    dev_path = "data/dev/"
    results_path = "temp_results"
    n_layers = 6
    n_heads = 6
    embed_size = 192
    mlp_hidden_size = embed_size * 4

    learning_rate = 5e-4
    gradient_clipping = 1.0
    weight_decay = None

    num_batches_to_train = 50000

    use_scheduler = False

    run_file_name = get_file_name(seq_len, batch_size, data_path, n_layers, n_heads, embed_size,
                                  mlp_hidden_size, learning_rate, gradient_clipping, weight_decay,
                                  num_batches_to_train, use_scheduler)
    loss_file_name = os.path.join(results_path, 'losses' + run_file_name + '.json')
    dev_loss_file_name = os.path.join(results_path, 'dev_losses' + run_file_name + '.json')
    sampling_file_name = os.path.join(results_path, 'sampling' + run_file_name + '.json')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer, tokenized_data = data.load_data(data_path)
    train_tokenizer, tokenized_train_data = data.load_data(train_path)
    # NOTE: are data items are longer by one than the sequence length,
    # They will be shortened by 1 when converted to training examples.
    train_iter = iter(data.RandomOrderDataIterator(tokenized_train_data, seq_len + 1))

    dev_tokenizer, tokenized_dev_data = data.load_data(dev_path)
    # NOTE: are data items are longer by one than the sequence length,
    # They will be shortened by 1 when converted to training examples.
    dev_iter = iter(data.RandomOrderDataIterator(tokenized_dev_data, seq_len + 1))

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
        model.parameters(), lr=learning_rate, betas=(0.9, 0.95))
    # scheduler = torch.optim.lr_scheduler.StepLR(
    #     optimizer, step_size=num_batches_to_train // 5, gamma=0.8
    # )

    model.train()

    num_batches = 0
    losses = []
    samples = []
    dev_losses = []
    while num_batches < num_batches_to_train:
        for batch in data.batch_items(train_iter, batch_size):
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
            # if use_scheduler:
                # scheduler.step()

            if num_batches % 10 == 0:
                print(f"Seen {num_batches} batches. last loss is: {loss.item()}")
                if num_batches % 100 == 0:
                    for _ in range(1):
                        model.eval()
                        # temperature( < 1) makes the model return more confident results (reducing entropy),
                        # while increasing the temperature ( > 1) makes the model return less confident results
                        # (increasing entropy).
                        sampled = tokenizer.detokenize(
                            model.better_sample_continuation(tokenizer.tokenize("Hello"),
                                                             500,
                                                             0.7,
                                                             5)
                        )
                        model.train()
                        print(f"Model sample: '''{sampled}'''")
                        samples.append(sampled)
                        if num_batches % 10000 == 0:
                            torch.save(model.state_dict(),
                                       os.path.join(results_path, f"llm_model_{num_batches}_{run_file_name}.pth"))
                    print("")
                    num_batches_to_eval = 20
                    num_dev_batches = 0
                    dev_loss = 0
                    for dev_batch in data.batch_items(dev_iter, batch_size):
                        if num_dev_batches >= num_batches_to_eval:
                            break
                        num_dev_batches += 1
                        dev_batch_x, dev_batch_y = lm.batch_to_labeled_samples(dev_batch)
                        logits = model(dev_batch_x.to(device))
                        dev_batch_loss = lm.compute_loss(logits, dev_batch_y.to(device))
                        dev_loss += dev_batch_loss.item()
                        print(f"Seen {num_dev_batches} dev batches. DEV last loss is: {dev_batch_loss.item()}")
                    dev_losses.append(dev_loss / num_dev_batches)
                    print(f"Seen {num_batches} batches. DEV last average loss is: {dev_loss / num_dev_batches}")
                    model.train()
    torch.save(model.state_dict(),
               os.path.join(results_path, f"llm_model_{num_batches}_{run_file_name}.pth"))
    plt.clf()
    plt.plot(losses)  # noqa
    plt.savefig(os.path.join(results_path, f"llm_losses_{run_file_name}.png"))
    with open(loss_file_name, "a") as output:
        json.dump(losses, output, indent=4)
    with open(sampling_file_name, mode="a", encoding='utf-8') as output:
        json.dump(samples, output, ensure_ascii=False, indent=4)
    with open(dev_loss_file_name, "a") as output:
        json.dump(dev_losses, output, indent=4)
    plt.clf()
    save_plot_of_loss_on_train_and_test(losses, dev_losses, results_path, run_file_name)


if __name__ == "__main__":
    main()
