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
    def __init__(self, config: GPTConfig):
        super().__init__(config)
        self.prefix_fc = torch.nn.Linear(2 * num_qubit, self.config.n_embd)

    def build_prefix(self, X1_np: np.ndarray, X2_np: np.ndarray, device: str = "cpu") -> torch.Tensor:
        X1 = torch.as_tensor(X1_np, dtype=torch.float32, device=device)
        X2 = torch.as_tensor(X2_np, dtype=torch.float32, device=device)
        z = torch.cat([X1, X2], dim=1)
        e = self.prefix_fc(z)
        prefix = e.unsqueeze(1).repeat(1, self.prefix_len, 1)
        return prefix

    def forward(self, idx, prefix_emb: torch.Tensor):
        device = idx.device
        b, t = idx.size()

        tok_emb = self.transformer.wte(idx)
        pos = torch.arange(0, self.prefix_len + t, dtype=torch.long, device=device)
        pos_emb = self.transformer.wpe(pos)

        x_prefix = prefix_emb + pos_emb[:self.prefix_len].unsqueeze(0)
        x_tokens = tok_emb + pos_emb[self.prefix_len:].unsqueeze(0)
        x = torch.cat((x_prefix, x_tokens), dim=1)

        x = self.transformer.drop(x)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def calculate_loss(self, tokens, energies, prefix_emb: torch.Tensor):
        current_tokens, next_tokens = tokens[:, :-1], tokens[:, 1:]
        logits = self(current_tokens, prefix_emb=prefix_emb)
        logits = logits[:, self.prefix_len:, :]

        next_token_mask = torch.nn.functional.one_hot(next_tokens, num_classes=self.config.vocab_size)
        next_token_logits = (logits * next_token_mask).sum(axis=2)
        total_logits = torch.sum(next_token_logits, dim=1)
        loss = torch.mean(torch.square(total_logits - energies.squeeze()))
        return loss

    @torch.no_grad()
    def generate(self, n_sequences, max_new_tokens, temperature=1.0, device="cpu", prefix_emb: torch.Tensor = None):
        idx = torch.zeros(size=(n_sequences, 1), dtype=torch.long, device=device)
        total_energy = torch.zeros((n_sequences, 1), device=device)

        for _ in range(max_new_tokens):
            max_tok_ctx = self.config.block_size - self.prefix_len
            idx_cond = idx if idx.size(1) <= max_tok_ctx else idx[:, -max_tok_ctx:]

            logits = self(idx_cond, prefix_emb=prefix_emb)
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
    probs = fidelity_circuit(X1, X2, ops)
    return float(abs(probs[0] - Y))


def get_sequence_energies(op_seq, X1, X2, Y, sample_indices, num_workers=4):
    inputs = [(ops, X1[i], X2[i], Y[i]) for ops, i in zip(op_seq, sample_indices)]
    with Pool(processes=num_workers) as pool:
        energies = pool.map(compute_energy_for_ops, inputs)
    return np.array(energies, dtype=np.float32).reshape(-1, 1)


def normalize_E(E, mu, sigma):
    return (E - mu) / sigma


def temperature(T_max, T_min, max_epoch, epoch):
    ratio = (T_min / T_max) ** (epoch / max_epoch)
    return T_max * ratio


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
    prefix_len = 8
    name = 'main_encode_prefix'

    # Save & Load
    resume = False
    resume_epoch = 5
    checkpoint_dir = f"{name}_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_name = f"{name}_checkpoint_{resume_epoch}.pt"

    X_train, _, Y_train, _ = data_load_and_process("kmnist", reduction_sz=num_qubit, train_len=population_size)

    gpt = GPTQE(GPTConfig(vocab_size=op_pool_size + 1, block_size=max_gate + prefix_len, dropout=0.2, bias=False))
    gpt.prefix_len = prefix_len
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
        X1, X2, Y, data_store = new_data(batch_size, X_train, Y_train)

        prefix_batch = gpt.build_prefix(X1, X2, device="cpu")

        S_total = train_size * 3
        S_ps = max(1, S_total // batch_size)
        n_gen = S_ps * batch_size
        prefix_for_gen = prefix_batch.detach().repeat_interleave(S_ps, dim=0)

        gpt.eval()
        train_token_seq_torch, _ = gpt.generate(
            n_sequences=n_gen,
            max_new_tokens=max_gate,
            temperature=temperature(T_max=T_max, T_min=T_min, max_epoch=max_epoch, epoch=i),
            device="cpu",
            prefix_emb=prefix_for_gen,
        )
        gpt.train()
        train_token_seq = train_token_seq_torch.numpy()
        train_op_inds = train_token_seq[:, 1:] - 1
        train_op_seq = op_pool[train_op_inds]

        sample_indices = np.repeat(np.arange(batch_size), S_ps)
        train_seq_en = get_sequence_energies(train_op_seq, X1, X2, Y, sample_indices)

        alpha = 0.1
        if mu is None:
            mu, sigma = float(train_seq_en.mean()), float(train_seq_en.std()) + 1e-8
        else:
            mu = alpha * float(train_seq_en.mean()) + (1 - alpha) * mu
            sigma = alpha * float(train_seq_en.std()) + (1 - alpha) * sigma
        print(f"[scale] μ={mu:.6f}, σ={sigma:.6f}")
        train_seq_en = normalize_E(train_seq_en, mu, sigma)

        train_token_seq, train_seq_en, sel_idx = select_token_and_en(train_token_seq, train_seq_en, train_size)
        sel_sample_idx = sample_indices[sel_idx]

        tokens = torch.from_numpy(train_token_seq)
        energies = torch.from_numpy(train_seq_en)
        sel_sample_idx_t = torch.from_numpy(sel_sample_idx.astype(np.int64))

        token_batches = torch.tensor_split(tokens, n_batches)
        energy_batches = torch.tensor_split(energies, n_batches)
        sample_idx_batches = torch.tensor_split(sel_sample_idx_t, n_batches)

        loss_record = 0
        losses = []
        for token_batch, energy_batch, si_batch in zip(token_batches, energy_batches, sample_idx_batches):
            opt.zero_grad()
            prefix_for_batch = gpt.build_prefix(X1[si_batch.numpy()], X2[si_batch.numpy()], device="cpu")
            loss = gpt.calculate_loss(token_batch, energy_batch, prefix_emb=prefix_for_batch)
            loss.backward()
            opt.step()
            loss_record += loss.item() / n_batches
        losses.append(loss_record)

        gpt.eval()
        S_eval_total = 100
        S_eval = max(1, S_eval_total // batch_size)
        n_eval = S_eval * batch_size
        prefix_for_eval = prefix_batch.detach().repeat_interleave(S_eval, dim=0)

        gen_token_seq, pred_Es = gpt.generate(
            n_sequences=n_eval,
            max_new_tokens=max_gate,
            temperature=0.01,
            device="cpu",
            prefix_emb=prefix_for_eval
        )
        pred_Es = pred_Es.numpy()
        print(gen_token_seq[0])

        gen_inds = (gen_token_seq[:, 1:] - 1).numpy()
        gen_op_seq = op_pool[gen_inds]
        sample_indices_eval = np.repeat(np.arange(batch_size), S_eval)
        true_Es = get_sequence_energies(gen_op_seq, X1, X2, Y, sample_indices_eval)

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
