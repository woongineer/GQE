from multiprocessing import Pool

import numpy as np
import pennylane as qml
import torch
from torch.nn import functional as F

from data import data_load_and_process, new_data_diffH
from model import GPT, GPTConfig
from utils import make_op_pool, apply_circuit, select_token_and_en, plot_result

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
        cumsum_logits = torch.cumsum(next_token_logits, dim=1)
        loss = torch.mean(torch.square(cumsum_logits - energies))
        return loss

    @torch.no_grad()
    def generate(self, n_sequences, max_new_tokens, temperature=1.0, device="cpu"):
        idx = torch.zeros(size=(n_sequences, 1), dtype=int, device=device)
        total_logits = torch.zeros(size=(n_sequences, 1), device=device)
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :]
            logits[:, 0] = float("inf")
            probs = F.softmax(-logits / temperature, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            total_logits += torch.gather(logits, index=idx_next, dim=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx, total_logits


@qml.qnode(dev, interface='torch')
def fidelity_circuit(x1, x2, circuit):
    apply_circuit(x1, circuit)
    qml.adjoint(apply_circuit)(x2, circuit)
    return qml.probs(wires=range(num_qubit))


def compute_all_subsequence_energies_for_ops(ops_and_data):
    ops, X1, X2, Y = ops_and_data

    sub_sequence_energies = []
    max_len = len(ops)

    for k in range(1, max_len + 1):
        sub_ops = ops[:k]

        energy_per_data_pair = []
        for single_x1, single_x2, single_y in zip(X1, X2, Y):
            probs = fidelity_circuit(single_x1, single_x2, sub_ops)
            es = -1 * (probs[0] - 0.5) * single_y
            energy_per_data_pair.append(es.item())

        sub_sequence_energies.append(np.mean(energy_per_data_pair))

    return sub_sequence_energies


def get_sequence_energies(op_seq, X1, X2, Y, num_workers=4):
    with Pool(processes=num_workers) as pool:
        inputs = [(ops, X1, X2, Y) for ops in op_seq]
        energies = pool.map(compute_all_subsequence_energies_for_ops, inputs)
    return np.array(energies, dtype=np.float32)


if __name__ == '__main__':
    population_size = 30
    batch_size = population_size
    max_epoch = 100
    gate_type = ['RX', 'RY', 'RZ', 'CNOT', 'H', 'I']
    op_pool = make_op_pool(gate_type=gate_type, num_qubit=num_qubit, num_param=num_qubit)
    op_pool_size = len(op_pool)
    train_size = 128
    n_batches = 8
    max_gate = 20

    X_train, _, Y_train, _ = data_load_and_process("kmnist", reduction_sz=num_qubit, train_len=population_size)

    gpt = GPTQE(GPTConfig(vocab_size=op_pool_size + 1, block_size=max_gate, dropout=0.2, bias=False))
    opt = gpt.configure_optimizers(weight_decay=0.01, learning_rate=5e-5, betas=(0.9, 0.999), device_type="cpu")
    gpt.train()
    pred_Es_t = []
    true_Es_t = []

    X1, X2, Y = new_data_diffH(batch_size, X_train, Y_train)
    fidelity_history = []
    loss_history = []
    for i in range(max_epoch):
        train_op_pool_inds = np.random.randint(op_pool_size, size=(train_size * 3, max_gate))
        train_op_seq = op_pool[train_op_pool_inds]
        train_token_seq = np.concatenate([np.zeros(shape=(train_size * 3, 1),
                                                   dtype=int), train_op_pool_inds + 1], axis=1)

        train_seq_en = get_sequence_energies(train_op_seq, X1, X2, Y)

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
        print(f"Iteration: {i + 1}, Loss: {losses[-1]}")

        # if i == 0 or (i + 1) % 10 == 0:
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
        true_Es = get_sequence_energies(gen_op_seq, X1, X2, Y)[:, -1].reshape(-1, 1)

        mae = np.mean(np.abs(pred_Es - true_Es))
        ave_E = np.mean(true_Es)

        pred_Es_t.append(pred_Es)
        true_Es_t.append(true_Es)

        print(f"Iter: {i + 1}, Loss: {losses[-1]}, MAE: {mae}, Ave True E: {ave_E}")

        gpt.train()
        fidelity_history.append(ave_E)
        loss_history.append(losses[-1])

    plot_result(fidelity_history, 'data_fix_diffH_snapshot_fidelity', 'data_fix_diffH_snapshot_fidelity.png')
    plot_result(loss_history, 'data_fix_diffH_snapshot_loss', 'data_fix_diffH_snapshot_loss.png')
