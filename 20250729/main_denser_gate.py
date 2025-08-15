import json
import os
import pickle
from multiprocessing import Pool

import numpy as np
import pennylane as qml
import torch

from data import data_load_and_process, new_data
from model import GPT, GPTConfig
from utils import select_token_and_en, plot_result, record_generated_results, \
    save_checkpoint, load_checkpoint, make_op_pool

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
        return logits

    def calculate_loss(self, tokens, energies):
        current_tokens, next_tokens = tokens[:, :-1], tokens[:, 1:]
        logits = self(current_tokens)
        next_token_mask = torch.nn.functional.one_hot(next_tokens, num_classes=self.config.vocab_size)
        next_token_logits = (logits * next_token_mask).sum(dim=2)

        scale_null = self.layout["scale_null"]
        mask = (next_tokens != scale_null)
        next_token_logits = next_token_logits * mask

        total_logits = next_token_logits.sum(dim=1)
        loss = torch.mean(torch.square(total_logits - energies.squeeze()))
        return loss

    def _sample_next_token_with_energy(self, current_seq, allowed_token_ids, bos_id, temperature):
        if current_seq.size(1) <= self.config.block_size:
            clipped_seq = current_seq
        else:
            clipped_seq = current_seq[:, -self.config.block_size:]

        logits = self(clipped_seq)[:, -1, :]

        # mask_forbidden = torch.ones_like(logits, dtype=torch.bool)
        # mask_forbidden[:, allowed_token_ids] = False
        # logits = logits.masked_fill(mask_forbidden, float('inf'))
        # logits[:, bos_id] = float('inf')
        scores = -logits / temperature
        scores_masked = torch.full_like(scores, -1e9)
        scores_masked[:, allowed_token_ids] = scores[:, allowed_token_ids]
        scores_masked[:, bos_id] = -1e9
        # tie-breaking
        scores_masked = scores_masked + 1e-6 * torch.randn_like(scores_masked)

        # probs = torch.softmax(-logits / temperature, dim=-1)
        probs = torch.softmax(scores_masked, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        picked_energy = torch.gather(logits, 1, next_token)
        return next_token, picked_energy

    # @torch.no_grad()
    # def generate(self, n_sequences, n_gate, scale_mask, temperature=1.0, device="cpu"):
    #     gate_ids = torch.arange(self.layout['gate_start'], self.layout['gate_end'] + 1, device=device)
    #     scale_ids = torch.arange(self.layout['scale_start'], self.layout['scale_end'] + 1, device=device)
    #     scale_null = self.layout['scale_null']
    #
    #     current_sequence = torch.zeros(size=(n_sequences, 1), dtype=torch.long, device=device)
    #     total_logit = torch.zeros((n_sequences, 1), device=device)
    #
    #     for _ in range(n_gate):
    #         next_gate_token, gate_energy = self._sample_next_token_with_energy(current_seq=current_sequence,
    #                                                                            allowed_token_ids=gate_ids,
    #                                                                            bos_id=self.layout['sequence_start'],
    #                                                                            temperature=temperature)
    #         current_sequence = torch.cat((current_sequence, next_gate_token), dim=1)
    #         total_logit += gate_energy
    #
    #         gate_idx_in_pool = (next_gate_token.squeeze(1) - self.layout["gate_start"]).numpy()
    #         is_scale_mask = torch.tensor(np.array(scale_mask)[gate_idx_in_pool], device=device, dtype=torch.bool)
    #
    #         if is_scale_mask.any():
    #             next_scale_token, scale_logit = self._sample_next_token_with_energy(current_seq=current_sequence,
    #                                                                                 allowed_token_ids=scale_ids,
    #                                                                                 bos_id=self.layout['sequence_start'],
    #                                                                                 temperature=temperature)
    #             next_scale_token = torch.where(is_scale_mask.unsqueeze(1), next_scale_token,
    #                                            torch.full_like(next_scale_token, scale_null))
    #             scale_logit = torch.where(is_scale_mask.unsqueeze(1), scale_logit, torch.zeros_like(scale_logit))
    #         else:
    #             next_scale_token = torch.full((n_sequences, 1), scale_null, device=device, dtype=torch.long)
    #             scale_logit = torch.zeros_like(next_scale_token, dtype=torch.float)
    #         current_sequence = torch.cat((current_sequence, next_scale_token), dim=1)
    #         total_logit += scale_logit
    #
    #     return current_sequence, total_logit

    ###########수정된 부분##########
    @torch.no_grad()
    def generate(self, n_sequences, n_gate, scale_mask, temperature=1.0, device="cpu"):
        gate_ids = torch.arange(self.layout['gate_start'], self.layout['gate_end'] + 1, device=device)
        scale_ids = torch.arange(self.layout['scale_start'], self.layout['scale_end'] + 1, device=device)
        scale_null = self.layout['scale_null']

        current_sequence = torch.zeros(size=(n_sequences, 1), dtype=torch.long, device=device)
        total_logit = torch.zeros((n_sequences, 1), device=device)

        for _ in range(n_gate):
            # 1) 게이트 샘플
            next_gate_token, gate_energy = self._sample_next_token_with_energy(
                current_seq=current_sequence,
                allowed_token_ids=gate_ids,
                bos_id=self.layout['sequence_start'],
                temperature=temperature
            )
            current_sequence = torch.cat((current_sequence, next_gate_token), dim=1)
            total_logit += gate_energy

            # 파라메트릭 여부 행별 판정
            g_idx_in_pool = (next_gate_token.squeeze(1) - self.layout["gate_start"]).cpu().numpy()
            is_param = torch.tensor(np.array(scale_mask)[g_idx_in_pool], device=device, dtype=torch.bool)

            # 2) 스케일 샘플: param 행만 scale_ids에서, 나머지는 NULL 고정
            next_scale_token = torch.full((n_sequences, 1), scale_null, dtype=torch.long, device=device)
            scale_logit = torch.zeros((n_sequences, 1), dtype=torch.float, device=device)

            param_idx = torch.nonzero(is_param, as_tuple=False).squeeze(1)
            if param_idx.numel() > 0:
                next_scale_param, scale_logit_param = self._sample_next_token_with_energy(
                    current_seq=current_sequence[param_idx],
                    allowed_token_ids=scale_ids,
                    bos_id=self.layout['sequence_start'],
                    temperature=temperature
                )
                next_scale_token[param_idx] = next_scale_param
                scale_logit[param_idx] = scale_logit_param

            current_sequence = torch.cat((current_sequence, next_scale_token), dim=1)
            total_logit += scale_logit

        return current_sequence, total_logit
    ###########수정된 부분##########


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


def apply_circuit(x, circuit):
    for gate in circuit:
        apply_gate(gate, x)


def apply_gate(gate, x):
    gate_type, param_idx, qubit_idx = gate
    ctrl_idx, target_idx = qubit_idx

    # gate 적용
    if gate_type == 'RX':
        qml.RX(x[param_idx[0]] * param_idx[1], wires=ctrl_idx)
    elif gate_type == 'RY':
        qml.RY(x[param_idx[0]] * param_idx[1], wires=ctrl_idx)
    elif gate_type == 'RZ':
        qml.RZ(x[param_idx[0]] * param_idx[1], wires=ctrl_idx)
    elif gate_type == 'H':
        qml.Hadamard(wires=ctrl_idx)
    elif gate_type == 'CNOT':
        qml.CNOT(wires=[ctrl_idx, target_idx])
    elif gate_type == 'I':
        qml.Identity(wires=ctrl_idx)
    elif gate_type == 'MultiRZ':
        qml.MultiRZ(x[param_idx[0]] * param_idx[1], wires=[ctrl_idx, target_idx])


def build_layout(gate_pool, scale_vals):
    sequence_start = 0
    gate_start = 1
    gate_end = gate_start + len(gate_pool) - 1
    scale_start = gate_end + 1
    scale_end = scale_start + len(scale_vals) - 1
    scale_null = scale_end + 1
    return {"sequence_start": sequence_start,
            "gate_start": gate_start, "gate_end": gate_end,
            "scale_start": scale_start, "scale_end": scale_end, "scale_null": scale_null}


def make_scale_mask(gate_pool):
    return [g[0] in ['RX', 'RY', 'RZ', 'MultiRZ'] for g in gate_pool]


def tokens_to_ops(tokens, gate_pool, layout, scale_vals):
    ops = []
    token_idx = 1
    total_tokens = len(tokens)
    while token_idx + 1 < total_tokens:
        gate_id = int(tokens[token_idx])
        scale_id = int(tokens[token_idx + 1])
        token_idx += 2

        if not (layout["gate_start"] <= gate_id <= layout["gate_end"]):
            break

        gate_info = gate_pool[gate_id - layout["gate_start"]]

        if gate_info[0] in ['RX', 'RY', 'RZ', 'MultiRZ']:
            if scale_id == layout["scale_null"]:
                continue
            scale_idx = scale_id - layout["scale_start"]
            ops.append((gate_info[0], (gate_info[1], float(scale_vals[scale_idx])), gate_info[2]))
        else:
            ops.append((gate_info[0], None, gate_info[2]))
    return ops

if __name__ == '__main__':
    population_size = 1000
    batch_size = 8
    max_epoch = 9

    scale_vals = np.array([0.1, 0.3, 0.5, 0.7, 1.0], dtype=float)
    gate_type = ['RX', 'RY', 'RZ', 'CNOT', 'H', 'I', 'MultiRZ']

    gate_pool = make_op_pool(gate_type=gate_type, num_qubit=num_qubit, num_param=num_qubit)
    layout = build_layout(gate_pool, scale_vals)
    scale_mask = make_scale_mask(gate_pool)
    vocab_size = layout["scale_null"] + 1
    train_size = 16
    n_batches = 8
    max_gate = 20  #28
    token_per_gate = 2

    block_size = max_gate * token_per_gate + 1

    T_max = 1000
    T_min = 0.04
    name = 'fix_sample_SM_denser_gate'

    # Save & Load
    resume = False
    resume_epoch = 5
    checkpoint_dir = f"{name}_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_name = f"{name}_checkpoint_{resume_epoch}.pt"

    X_train, _, Y_train, _ = data_load_and_process("kmnist", reduction_sz=num_qubit, train_len=population_size)

    gpt = GPTQE(GPTConfig(vocab_size=vocab_size, block_size=block_size, dropout=0.2, bias=False))
    gpt.layout = layout
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
        X1, X2, Y, data_store = new_data(batch_size, X_train, Y_train)
        with open(f"{name}_data_store.pkl", "wb") as f:
            pickle.dump(data_store, f)

    for i in range(start_epoch, max_epoch):
        gpt.eval()
        train_token_seq_torch, _ = gpt.generate(
            n_sequences=train_size * 3,
            scale_mask=scale_mask,
            n_gate=max_gate,
            temperature=temperature(T_max=T_max, T_min=T_min, max_epoch=max_epoch, epoch=i),
            device="cpu",
        )
        gpt.train()
        train_token_seq = train_token_seq_torch.numpy()
        train_op_seq = []
        for tokens in train_token_seq:
            ops = tokens_to_ops(tokens, gate_pool, layout, scale_vals)
            train_op_seq.append(ops)

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
            scale_mask=scale_mask,
            n_gate=max_gate,
            temperature=0.01,
            device="cpu"
        )
        pred_Es = pred_Es.numpy()
        print(gen_token_seq[0])

        gen_op_seq = []
        for seq in gen_token_seq.numpy():
            ops = tokens_to_ops(seq.tolist(), gate_pool, layout, scale_vals)
            gen_op_seq.append(ops)
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
