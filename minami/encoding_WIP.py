import json
from multiprocessing import Pool

import numpy as np
import pennylane as qml
import torch
from torch.nn import functional as F

from custom.data import data_load_and_process, new_data
from custom.model import GPT, GPTConfig
from custom.utils import make_op_pool, apply_circuit, select_token_and_en, plot_result, record_generated_results

num_qubit = 4
dev = qml.device("default.qubit", wires=num_qubit)


class GPTQE(GPT):
    def forward(self, idx):
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
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
        total_logp = torch.zeros((n_sequences, 1), device=device)
        total_energy = torch.zeros((n_sequences, 1), device=device)

        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :]
            logits[:, 0] = float("inf")

            log_probs = F.log_softmax(-logits / temperature, dim=-1)
            probs = log_probs.exp()

            idx_next = torch.multinomial(probs, num_samples=1)

            total_logp += torch.gather(log_probs, 1, idx_next)
            total_energy += torch.gather(logits, 1, idx_next)

            idx = torch.cat((idx, idx_next), dim=1)

        return idx, total_logp, total_energy


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


class Encoder(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.linear1 = torch.nn.Linear(in_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, out_dim)

    def forward(self, x):  # 예시: x는 (B, F)
        return self.linear2(torch.relu(self.linear1(x)))


if __name__ == '__main__':
    population_size = 8
    batch_size = population_size
    max_epoch = 100
    gate_type = ['RX', 'RY', 'RZ', 'CNOT', 'H', 'I']
    op_pool = make_op_pool(gate_type=gate_type, num_qubit=num_qubit, num_param=num_qubit)
    op_pool_size = len(op_pool)
    train_size = 32
    n_batches = 8
    max_gate = 20
    T_max = 1000
    T_min = 0.01


    X_train, _, Y_train, _ = data_load_and_process("kmnist", reduction_sz=num_qubit, train_len=population_size)

    gpt = GPTQE(GPTConfig(vocab_size=op_pool_size + 1, block_size=max_gate, dropout=0.2, bias=False))
    opt = gpt.configure_optimizers(weight_decay=0.01, learning_rate=5e-5, betas=(0.9, 0.999), device_type="cpu")
    gpt.train()
    pred_Es_t = []
    true_Es_t = []

    X1, X2, Y = new_data(batch_size, X_train, Y_train)
    mu, sigma = None, None

    fidelity_history = []
    loss_history = []
    all_gen_records = []
    for i in range(max_epoch):
        gpt.eval()
        train_token_seq_torch, _, _ = gpt.generate(
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

        # if i == 0 or (i + 1) % 10 == 0:
        gpt.eval()
        gen_token_seq, _, pred_Es = gpt.generate(
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

        mae = np.mean(np.abs(pred_Es - true_Es_norm))
        ave_E = np.mean(true_Es)

        pred_Es_t.append(pred_Es)
        true_Es_t.append(true_Es)

        print(f"Iter: {i + 1}, Loss: {losses[-1]}, MAE: {mae}, Ave True E: {ave_E}")

        fidelity_history.append(ave_E)
        loss_history.append(losses[-1])
        record_generated_results(all_gen_records, i + 1, gen_op_seq, true_Es)

    plot_result(fidelity_history, 'data_fix_sampling_fidelity', 'data_fix_smapling_fidelity.png')
    plot_result(loss_history, 'data_fix_sampling_loss', 'data_fix_sampling_loss.png')
    with open("../custom/result/data_fix_sampling_generated_circuit.json", "w") as f:
        json.dump(all_gen_records, f, indent=2)
