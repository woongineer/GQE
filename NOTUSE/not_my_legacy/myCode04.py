import os

import numpy as np
from multiprocessing import Pool
import pennylane as qml
import torch
from torch.nn import functional as F

from data import data_load_and_process, get_class_balanced_batch
from model import GPT, GPTConfig

num_qubit = 4
dev = qml.device("default.qubit", wires=num_qubit)


def make_op_pool(gate_type, num_qubit, num_param):
    op_pool = []

    for gate in gate_type:
        if gate in ['RX', 'RY', 'RZ']:
            for q in range(num_qubit):
                for p in range(num_param):
                    op_pool.append((gate, p, (q, None)))
        elif gate in ['H', 'I']:
            for q in range(num_qubit):
                op_pool.append((gate, None, (q, None)))
        elif gate == 'CNOT':
            for control in range(num_qubit):
                for target in range(num_qubit):
                    if control != target:
                        op_pool.append((gate, None, (control, target)))

    return np.array(op_pool, dtype=object)


class GPTQE(GPT):
    def forward(self, idx):
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def calculate_loss(self, tokens, energies):
        current_tokens, next_tokens = tokens[:, :-1], tokens[:, 1:]
        # calculate the logits for the next possible tokens in the sequence
        logits = self(current_tokens)
        # get the logit for the actual next token in the sequence
        next_token_mask = torch.nn.functional.one_hot(
            next_tokens, num_classes=self.config.vocab_size
        )
        next_token_logits = (logits * next_token_mask).sum(axis=2)
        # calculate the cumulative logits for each subsequence
        cumsum_logits = torch.cumsum(next_token_logits, dim=1)
        # match cumulative logits to subsequence energies
        loss = torch.mean(torch.square(cumsum_logits - energies))
        return loss

    @torch.no_grad()
    def generate(self, n_sequences, max_new_tokens, temperature=1.0, device="cpu"):
        idx = torch.zeros(size=(n_sequences, 1), dtype=int, device=device)
        total_logits = torch.zeros(size=(n_sequences, 1), device=device)
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits = self(idx_cond)
            # pluck the logits at the final step
            logits = logits[:, -1, :]
            # set the logit of the first token so that its probability will be zero
            logits[:, 0] = float("inf")
            # apply softmax to convert logits to (normalized) probabilities and scale by desired temperature
            probs = F.softmax(-logits / temperature, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # # Accumulate logits
            total_logits += torch.gather(logits, index=idx_next, dim=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
        return idx, total_logits


def apply_gate(gate, x):
    gate_type, param_idx, qubit_idx = gate
    ctrl_idx, target_idx = qubit_idx

    # gate 적용
    if gate_type == 'RX':
        qml.RX(x[param_idx], wires=ctrl_idx)
    elif gate_type == 'RY':
        qml.RY(x[param_idx], wires=ctrl_idx)
    elif gate_type == 'RZ':
        qml.RZ(x[param_idx], wires=ctrl_idx)
    elif gate_type == 'H':
        qml.Hadamard(wires=ctrl_idx)
    elif gate_type == 'CNOT':
        qml.CNOT(wires=[ctrl_idx, target_idx])
    elif gate_type == 'I':
        qml.Identity(wires=ctrl_idx)


def apply_circuit(x, circuit):
    for gate in circuit:
        apply_gate(gate, x)


@qml.qnode(dev, interface='torch')
def fidelity_circuit(x1, x2, circuit):
    apply_circuit(x1, circuit)
    qml.adjoint(apply_circuit)(x2, circuit)
    return qml.probs(wires=range(4))


# def get_sequence_energies(op_seq, X1, X2, Y):
#     energies = []
#     for i, ops in enumerate(op_seq):
#         print(i)
#         energy_per_seq = []
#         for single_x1, single_x2, single_y in zip(X1, X2, Y):
#             probs = fidelity_circuit(single_x1, single_x2, ops)
#             es = probs[0] * single_y
#             energy_per_seq.append(es.item())
#         energy_per_seq_average = np.mean(energy_per_seq)
#         energies.append(energy_per_seq_average)
#
#     return np.array(energies)


def compute_energy_for_ops(ops_and_data):
    ops, X1, X2, Y = ops_and_data
    energy_per_seq = []
    for single_x1, single_x2, single_y in zip(X1, X2, Y):
        probs = fidelity_circuit(single_x1, single_x2, ops)
        es = -1 * probs[0] * single_y
        energy_per_seq.append(es.item())
    return np.mean(energy_per_seq)

def get_sequence_energies(op_seq, X1, X2, Y, num_workers=4):
    with Pool(processes=num_workers) as pool:
        inputs = [(ops, X1, X2, Y) for ops in op_seq]
        energies = pool.map(compute_energy_for_ops, inputs)
    return np.array(energies, dtype=np.float32).reshape(-1, 1)


def evaluate_fidelity_of_generated_circuits(gpt, op_pool, X1, X2, Y, max_gate, device="cpu"):
    gpt.eval()
    gen_token_seq, _ = gpt.generate(
        n_sequences=100,
        max_new_tokens=max_gate,
        temperature=0.001,
        device=device
    )
    gen_inds = (gen_token_seq[:, 1:] - 1).cpu().numpy()
    gen_op_seq = op_pool[gen_inds]

    # 직접 fidelity-based energy 계산
    true_Es = get_sequence_energies(gen_op_seq, X1, X2, Y)[:, -1].reshape(-1, 1)
    ave_energy = np.mean(true_Es)

    print(f"[Fidelity Evaluation] Avg Energy of Generated Circuits: {ave_energy:.4f}")
    return ave_energy



if __name__ == '__main__':
    population_size = 10000
    batch_size = 50
    max_epoch = population_size // batch_size
    gate_type = ['RX', 'RY', 'RZ', 'CNOT', 'H', 'I']
    op_pool = make_op_pool(gate_type=gate_type, num_qubit=num_qubit, num_param=num_qubit)
    op_pool_size = len(op_pool)
    train_size = 256
    n_batches = 8
    max_gate = 20

    X_train, _, Y_train, _ = data_load_and_process("kmnist", reduction_sz=num_qubit, train_len=population_size)

    gpt = GPTQE(GPTConfig(
        vocab_size=op_pool_size + 1,
        block_size=max_gate,
        dropout=0.2,
        bias=False
    )).to("cpu")
    opt = gpt.configure_optimizers(
        weight_decay=0.01, learning_rate=5e-5, betas=(0.9, 0.999), device_type="cpu"
    )
    gpt.train()
    pred_Es_t = []
    true_Es_t = []

    for i in range(max_epoch * 3):
        X1, X2, Y = get_class_balanced_batch(batch_size, X_train, Y_train)

        train_op_pool_inds = np.random.randint(op_pool_size, size=(train_size, max_gate))
        train_op_seq = op_pool[train_op_pool_inds]
        train_token_seq = np.concatenate([
            np.zeros(shape=(train_size, 1), dtype=int),  # starting token is 0
            train_op_pool_inds + 1  # shift operator inds by one
        ], axis=1)
        train_seq_en = get_sequence_energies(train_op_seq, X1, X2, Y)

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

        if (i + 1) % 10 == 0:
            # For GPT evaluation
            gpt.eval()
            gen_token_seq, pred_Es = gpt.generate(
                n_sequences=100,
                max_new_tokens=max_gate,
                temperature=0.001,  # Use a low temperature to emphasize the difference in logits
                device="cpu"
            )
            pred_Es = pred_Es.cpu().numpy()

            gen_inds = (gen_token_seq[:, 1:] - 1).cpu().numpy()
            gen_op_seq = op_pool[gen_inds]
            true_Es = get_sequence_energies(gen_op_seq, X1, X2, Y)[:, -1].reshape(-1, 1)

            mae = np.mean(np.abs(pred_Es - true_Es))
            ave_E = np.mean(true_Es)

            pred_Es_t.append(pred_Es)
            true_Es_t.append(true_Es)

            print(f"Iteration: {i + 1}, Loss: {losses[-1]}, MAE: {mae}, Ave E: {ave_E}")

            fidelity_score = evaluate_fidelity_of_generated_circuits(gpt, op_pool, X1, X2, Y, max_gate)
            print(f"Fidelity Score: {fidelity_score}")

            gpt.train()
