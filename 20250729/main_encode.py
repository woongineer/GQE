import json
import os
from multiprocessing import Pool

import numpy as np
import pennylane as qml
import torch
from torch.nn import functional as F

from data import data_load_and_process, new_data
from model import GPT, GPTConfig
from utils import make_op_pool, apply_circuit, select_token_and_en, plot_result, record_generated_results, \
    save_checkpoint, load_checkpoint

num_qubit = 4
dev = qml.device("default.qubit", wires=num_qubit)


class GPTQE(GPT):
    def forward(self, idx):
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        logits = logits + self.data_bias
        return logits

    def calculate_loss(self, tokens, energies):
        current_tokens, next_tokens = tokens[:, :-1], tokens[:, 1:]
        logits = self(current_tokens)
        next_token_mask = torch.nn.functional.one_hot(next_tokens, num_classes=self.config.vocab_size)
        next_token_logits = (logits * next_token_mask).sum(axis=2)
        total_logits = torch.sum(next_token_logits, dim=1)
        loss = torch.mean(torch.square(total_logits - energies.squeeze()))
        return loss

    @torch.no_grad()
    def generate(self, n_sequences, max_new_tokens, temperature=1.0, device="cpu"):
        idx = torch.zeros(size=(n_sequences, 1), dtype=int, device=device)
        total_energy = torch.zeros((n_sequences, 1), device=device)

        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :]
            logits[:, 0] = float("inf")

            log_probs = F.log_softmax(-logits / temperature, dim=-1)
            probs = log_probs.exp()

            idx_next = torch.multinomial(probs, num_samples=1)

            total_energy += torch.gather(logits, 1, idx_next)

            idx = torch.cat((idx, idx_next), dim=1)

        return idx, total_energy


@qml.qnode(dev, interface='torch')
def fidelity_circuit(x1, x2, circuit):
    apply_circuit(x1, circuit)
    qml.adjoint(apply_circuit)(x2, circuit)
    return qml.probs(wires=range(num_qubit))


def compute_energy_for_ops(ops_and_data):
    ops, X1, X2, Y = ops_and_data
    energy_per_seq = []
    for single_x1, single_x2, single_y in zip(X1, X2, Y):
        probs = fidelity_circuit(single_x1, single_x2, ops)
        es = abs(probs[0] - single_y)
        energy_per_seq.append(es.item())
    return np.mean(energy_per_seq)


def get_sequence_energies(op_seq, X1, X2, Y, num_workers=4):
    with Pool(processes=num_workers) as pool:
        inputs = [(ops, X1, X2, Y) for ops in op_seq]
        energies = pool.map(compute_energy_for_ops, inputs)
    return np.array(energies, dtype=np.float32).reshape(-1, 1)


def normalize_E(E, mu, sigma):
    return (E - mu) / sigma


def temperature(T_max, T_min, max_epoch, epoch):
    ratio = (T_min / T_max) ** (epoch / max_epoch)
    return T_max * ratio


def circular_feature(X):
    feature = []
    for k in range(1, 3):
        feature.append(torch.cos(k * X.reshape(-1)))
        feature.append(torch.sin(k * X.reshape(-1)))
    return torch.cat(feature, dim=0)


def batch_representation(X1, X2, vocab_size, scale=0.3):
    parts = [circular_feature(X1), circular_feature(X2)]
    parts.append(circular_feature(X1 - X2))

    phi = torch.cat(parts, dim=0)
    phi = phi - phi.mean()
    std = phi.std().clamp_min(1e-6)
    phi = phi / std

    pad = (vocab_size - (len(phi) % vocab_size)) % vocab_size
    phi = torch.nn.functional.pad(phi, (0, pad))
    folded = phi.view(vocab_size, -1).sum(dim=1)

    folded = torch.tanh(folded) * scale
    folded[0] = 0.0

    return folded.view(1, 1, -1)



if __name__ == '__main__':
    population_size = 1000
    batch_size = 8
    max_epoch = 9
    gate_type = ['RX', 'RY', 'RZ', 'CNOT', 'H', 'I']
    op_pool = make_op_pool(gate_type=gate_type, num_qubit=num_qubit, num_param=num_qubit)
    op_pool_size = len(op_pool)
    train_size = 16
    n_batches = 8
    max_gate = 56
    T_max = 1000
    T_min = 0.04
    name = 'fix_sample_SM'

    # Save & Load
    resume = False
    resume_epoch = 5
    checkpoint_dir = f"{name}_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_name = f"{name}_checkpoint_{resume_epoch}.pt"

    X_train, _, Y_train, _ = data_load_and_process("kmnist", reduction_sz=num_qubit, train_len=population_size)

    gpt = GPTQE(GPTConfig(vocab_size=op_pool_size + 1, block_size=max_gate, dropout=0.2, bias=False))
    gpt.data_bias = torch.zeros(1, 1, op_pool_size + 1)
    opt = gpt.configure_optimizers(weight_decay=0.01, learning_rate=5e-5, betas=(0.9, 0.999), device_type="cpu")
    gpt.train()

    # Save & Load
    if resume:
        print("Resuming from checkpoint...")
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        start_epoch, mu, sigma, seed, loss_history, fidelity_history, all_gen_records, X1, X2, Y = \
            load_checkpoint(gpt, opt, checkpoint_path)
        torch.manual_seed(seed)
        np.random.seed(seed)
    else:
        start_epoch = 0
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        mu, sigma = None, None
        fidelity_history = []
        loss_history = []
        all_gen_records = []

    for i in range(start_epoch, max_epoch):
        X1, X2, Y, _ = new_data(batch_size, X_train, Y_train)
        gpt.data_bias = batch_representation(X1, X2, op_pool_size + 1)

        gpt.eval()
        train_token_seq_torch, _ = gpt.generate(
            n_sequences=train_size * 3,
            max_new_tokens=max_gate,
            temperature=temperature(T_max=T_max, T_min=T_min, max_epoch=max_epoch, epoch=i),
            device="cpu",
        )
        gpt.train()
        train_token_seq = train_token_seq_torch.numpy()
        train_op_inds = train_token_seq[:, 1:] - 1
        train_op_seq = op_pool[train_op_inds]

        train_seq_en = get_sequence_energies(train_op_seq, X1, X2, Y)

        alpha = 0.1
        if mu is None:
            mu, sigma = float(train_seq_en.mean()), float(train_seq_en.std()) + 1e-8
        else:
            mu = alpha * float(train_seq_en.mean()) + (1 - alpha) * mu
            sigma = alpha * float(train_seq_en.std()) + (1 - alpha) * sigma
        print(f"[scale] μ={mu:.6f}, σ={sigma:.6f}")
        train_seq_en = normalize_E(train_seq_en, mu, sigma)

        train_token_seq, train_seq_en = select_token_and_en(train_token_seq, train_seq_en, train_size)

        tokens = torch.from_numpy(train_token_seq)
        energies = torch.from_numpy(train_seq_en)
        train_inds = np.arange(train_size)
        token_batches = torch.tensor_split(tokens[train_inds], n_batches)
        energy_batches = torch.tensor_split(energies[train_inds], n_batches)

        loss_record = 0
        losses = []
        for token_batch, energy_batch in zip(token_batches, energy_batches):
            opt.zero_grad()
            loss = gpt.calculate_loss(token_batch, energy_batch)
            loss.backward()
            opt.step()
            loss_record += loss.item() / n_batches
        losses.append(loss_record)

        gpt.eval()
        gen_token_seq, pred_Es = gpt.generate(
            n_sequences=100,
            max_new_tokens=max_gate,
            temperature=0.01,
            device="cpu"
        )
        pred_Es = pred_Es.numpy()
        print(gen_token_seq[0])

        gen_inds = (gen_token_seq[:, 1:] - 1).numpy()
        gen_op_seq = op_pool[gen_inds]
        true_Es = get_sequence_energies(gen_op_seq, X1, X2, Y)
        true_Es_norm = normalize_E(true_Es, mu, sigma)

        ave_pred_E = np.mean(pred_Es)
        ave_E = np.mean(true_Es)

        print(f"Iter: {i + 1}, Loss: {losses[-1]}, Ave Pred E: {ave_pred_E}, Ave True E: {ave_E}")

        fidelity_history.append(ave_E)
        loss_history.append(losses[-1])
        record_generated_results(all_gen_records, i + 1, gen_op_seq, true_Es)

        # Save & Load
        checkpoint_path = os.path.join(checkpoint_dir, f"{name}_checkpoint_{i + 1}.pt")
        save_checkpoint(
            model=gpt,
            optimizer=opt,
            epoch=i + 1,
            mu=mu,
            sigma=sigma,
            seed=seed,
            loss_hist=loss_history,
            fidelity_hist=fidelity_history,
            records=all_gen_records,
            X1=X1,
            X2=X2,
            Y=Y,
            path=checkpoint_path,
        )

    plot_result(fidelity_history, f'{name}_fidelity', f'{name}_fidelity.png')
    plot_result(loss_history, f'{name}_loss', f'{name}_loss.png')
    with open(f"{name}_generated_circuit.json", "w") as f:
        json.dump(all_gen_records, f, indent=2)
