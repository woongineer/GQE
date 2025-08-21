import json
import os
import pickle
from multiprocessing import Pool

import numpy as np
import pennylane as qml
import torch
from torch.nn import functional as F

from data import data_load_and_process, new_data
from model import GPT, GPTConfig
from utils import select_token_and_en, plot_result, record_generated_results, \
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
        return logits

    def calculate_loss(self, tokens, energies):
        current_tokens, next_tokens = tokens[:, :-1], tokens[:, 1:]
        logits = self(current_tokens)
        next_token_mask = torch.nn.functional.one_hot(next_tokens, num_classes=self.config.vocab_size)
        next_token_logits = (logits * next_token_mask).sum(dim=2)

        if hasattr(self, "E_NULL_ID") and hasattr(self, "M_NULL_ID"):
            mask = (next_tokens != self.E_NULL_ID) & (next_tokens != self.M_NULL_ID)
            next_token_logits = next_token_logits * mask

        total_logits = next_token_logits.sum(dim=1)
        loss = torch.mean(torch.square(total_logits - energies.squeeze()))
        return loss


    @torch.no_grad()
    def generate(self, n_sequences, n_gates, temperature=1.0, device="cpu"):
        layout = self.layout
        gate_token_ids = torch.arange(layout["gate_start"], layout["gate_end"] + 1, device=device)
        E_token_ids = torch.arange(layout["E_start"], layout["E_end"] + 1, device=device)
        M_token_ids = torch.arange(layout["M_start"], layout["M_end"] + 1, device=device)
        BOS_TOKEN_ID = layout["BOS_ID"]
        E_NULL_TOKEN = layout["E_NULL"]
        M_NULL_TOKEN = layout["M_NULL"]

        current_sequence = torch.full((n_sequences, 1), BOS_TOKEN_ID, dtype=torch.long, device=device)
        total_energy_logit = torch.zeros((n_sequences, 1), device=device)

        def step_sample(allowed_token_ids):
            # block_size를 넘으면 뒤에서부터 자르기
            clipped_seq = current_sequence if current_sequence.size(1) <= self.config.block_size else current_sequence[:,-self.config.block_size:]

            # 모델 출력에서 마지막 토큰 위치의 로짓 추출
            logits = self(clipped_seq)[:, -1, :]  # (B, vocab_size)

            # 모든 토큰 금지(True)로 시작해서 allowed_token_ids만 허용(False)
            mask_forbidden = torch.ones_like(logits, dtype=torch.bool)
            mask_forbidden[:, allowed_token_ids] = False

            # 금지된 토큰을 +inf로 채워 softmax에서 확률 0이 되게 함
            logits = logits.masked_fill(mask_forbidden, float('inf'))

            # BOS 토큰은 항상 금지
            logits[:, BOS_TOKEN_ID] = float('inf')

            # 낮은 로짓일수록 높은 확률이 되도록 부호 반전 후 softmax
            log_probs = torch.nn.functional.log_softmax(-logits / temperature, dim=-1)
            probs = log_probs.exp()

            # 확률분포에서 토큰 샘플링
            next_token = torch.multinomial(probs, num_samples=1)  # (B,1)

            # 샘플된 토큰의 원래 로짓(energy)을 추출
            picked_energy_logit = torch.gather(logits, 1, next_token)  # (B,1)
            return next_token, picked_energy_logit

        for _ in range(n_gates):
            # 1) 게이트 토큰 샘플링
            next_gate_token, gate_energy = step_sample(gate_token_ids)
            current_sequence = torch.cat((current_sequence, next_gate_token), dim=1)
            total_energy_logit += gate_energy

            # 파라메트릭 게이트인지 여부 판단
            gate_index_in_pool = (next_gate_token.squeeze(1) - layout["gate_start"]).cpu().numpy()
            is_param_gate_mask = torch.tensor(self.param_gate_mask[gate_index_in_pool], device=device, dtype=torch.bool)

            # 2) E 토큰 샘플링
            if is_param_gate_mask.any():
                next_E_token, E_energy = step_sample(E_token_ids)
                next_E_token = torch.where(is_param_gate_mask.unsqueeze(1), next_E_token,
                                           torch.full_like(next_E_token, E_NULL_TOKEN))
                E_energy = torch.where(is_param_gate_mask.unsqueeze(1), E_energy, torch.zeros_like(E_energy))
            else:
                next_E_token = torch.full((n_sequences, 1), E_NULL_TOKEN, device=device, dtype=torch.long)
                E_energy = torch.zeros_like(next_E_token, dtype=torch.float)
            current_sequence = torch.cat((current_sequence, next_E_token), dim=1)
            total_energy_logit += E_energy

            # 3) M 토큰 샘플링
            if is_param_gate_mask.any():
                next_M_token, M_energy = step_sample(M_token_ids)
                next_M_token = torch.where(is_param_gate_mask.unsqueeze(1), next_M_token,
                                           torch.full_like(next_M_token, M_NULL_TOKEN))
                M_energy = torch.where(is_param_gate_mask.unsqueeze(1), M_energy, torch.zeros_like(M_energy))
            else:
                next_M_token = torch.full((n_sequences, 1), M_NULL_TOKEN, device=device, dtype=torch.long)
                M_energy = torch.zeros_like(next_M_token, dtype=torch.float)
            current_sequence = torch.cat((current_sequence, next_M_token), dim=1)
            total_energy_logit += M_energy

        return current_sequence, total_energy_logit


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


def make_op_pool(gate_type, num_qubit, num_param):
    op_pool = []

    for gate in gate_type:
        if gate in ['RX', 'RY', 'RZ']:
            for q in range(num_qubit):
                for p in range(num_param):
                    op_pool.append((gate, p, (q, None)))
        elif gate == 'MultiRZ':
            for q1 in range(num_qubit):
                for q2 in range(q1 + 1, num_qubit):
                    for p in range(num_param):
                        op_pool.append((gate, p, (q1, q2)))
        elif gate in ['H', 'I']:
            for q in range(num_qubit):
                op_pool.append((gate, None, (q, None)))
        elif gate == 'CNOT':
            for control in range(num_qubit):
                for target in range(num_qubit):
                    if control != target:
                        op_pool.append((gate, None, (control, target)))

    return np.array(op_pool, dtype=object)


def is_param_gate(g):
    return g[0] in ['RX', 'RY', 'RZ', 'MultiRZ']


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


def build_layout(gate_pool, E_vals, M_vals):
    BOS_ID = 0
    gate_start = 1
    gate_end   = gate_start + len(gate_pool) - 1
    E_start    = gate_end + 1
    E_end      = E_start + len(E_vals) - 1
    E_NULL     = E_end + 1
    M_start    = E_NULL + 1
    M_end      = M_start + len(M_vals) - 1
    M_NULL     = M_end + 1
    vocab_size = M_NULL + 1
    return {
        "BOS_ID": BOS_ID,
        "gate_start": gate_start, "gate_end": gate_end,
        "E_start": E_start, "E_end": E_end, "E_NULL": E_NULL,
        "M_start": M_start, "M_end": M_end, "M_NULL": M_NULL,
        "vocab_size": vocab_size,
    }


def tokens_to_opseq(tokens_1d, gate_pool, layout, E_vals, M_vals):
    ops = []
    i = 1  # [0(BOS), g, e, m, g, e, m, ...]
    T = len(tokens_1d)
    while i + 2 < T:
        g_id = int(tokens_1d[i]); e_id = int(tokens_1d[i+1]); m_id = int(tokens_1d[i+2])
        i += 3
        if not (layout["gate_start"] <= g_id <= layout["gate_end"]):
            break
        gate = gate_pool[g_id - layout["gate_start"]]
        if is_param_gate(gate):
            if e_id == layout["E_NULL"] or m_id == layout["M_NULL"]:
                continue  # 문법 위반 방어
            e_idx = e_id - layout["E_start"]
            m_idx = m_id - layout["M_start"]
            s = (2.0 ** float(E_vals[e_idx])) * float(M_vals[m_idx])
            p = gate[1]
            ops.append((gate[0], (p, s), gate[2]))
        else:
            ops.append((gate[0], None, gate[2]))
    return ops


if __name__ == '__main__':
    population_size = 1000
    batch_size = 8
    max_epoch = 20

    E_vals = np.array([-2, -1, 0, 1], dtype=float)
    M_vals = np.array([0.70, 0.85, 1.00, 1.15], dtype=float)

    gate_type = ['RX', 'RY', 'RZ', 'CNOT', 'H', 'I', 'MultiRZ']
    gate_pool = make_op_pool(gate_type=gate_type, num_qubit=num_qubit, num_param=num_qubit)
    layout = build_layout(gate_pool, E_vals, M_vals)
    op_pool_size = len(gate_pool)
    param_gate_mask = np.array([is_param_gate(g) for g in gate_pool], dtype=bool)

    train_size = 16
    n_batches = 8
    max_gate = 20
    tokens_per_gate = 3
    block_size = max_gate * tokens_per_gate + 1

    T_max = 1000
    T_min = 0.04
    name = 'fix_sample_SM_more_gate_denser'

    # Save & Load
    resume = False
    resume_epoch = 5
    checkpoint_dir = f"{name}_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_name = f"{name}_checkpoint_{resume_epoch}.pt"

    X_train, _, Y_train, _ = data_load_and_process("kmnist", reduction_sz=num_qubit, train_len=population_size)

    total_vocab_size = layout["vocab_size"]
    gpt = GPTQE(GPTConfig(vocab_size=total_vocab_size, block_size=block_size, dropout=0.2, bias=False))
    gpt.layout = layout
    gpt.gate_pool = gate_pool
    gpt.param_gate_mask = param_gate_mask
    gpt.E_NULL_ID = layout["E_NULL"]
    gpt.M_NULL_ID = layout["M_NULL"]

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
            n_gates=max_gate,
            temperature=temperature(T_max=T_max, T_min=T_min, max_epoch=max_epoch, epoch=i),
            device="cpu",
        )
        gpt.train()
        train_token_seq = train_token_seq_torch.numpy()
        train_op_seq = []
        for seq in train_token_seq:
            ops = tokens_to_opseq(seq.tolist(), gate_pool, layout, E_vals, M_vals)
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

        train_token_seq, train_seq_en, _ = select_token_and_en(train_token_seq, train_seq_en, train_size)

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
            n_gates=max_gate,
            temperature=0.01,
            device="cpu"
        )
        pred_Es = pred_Es.numpy()
        print(gen_token_seq[0])

        gen_op_seq = []
        for seq in gen_token_seq.numpy():
            ops = tokens_to_opseq(seq.tolist(), gate_pool, layout, E_vals, M_vals)
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
