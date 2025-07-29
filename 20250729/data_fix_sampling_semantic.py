import json
from multiprocessing import Pool

import numpy as np
import pennylane as qml
import torch
from torch.nn import functional as F
import pickle

from NOTUSE.my_legacy_data_fix_py_is_first_hope.data import data_load_and_process, new_data
from NOTUSE.my_legacy_data_fix_py_is_first_hope.semantic_model import GPT, GPTConfig
from NOTUSE.my_legacy_data_fix_py_is_first_hope.utils import make_op_pool, apply_circuit, select_token_and_en, plot_result, record_generated_results

num_qubit = 4
dev = qml.device("default.qubit", wires=num_qubit)

GATE_LIST = ['RX', 'RY', 'RZ', 'CNOT', 'H', 'I']

def indices_to_gate_tuple(g_idx, q_idx, p_idx, t_idx):
    gate = GATE_LIST[g_idx]
    if gate in ['RX', 'RY', 'RZ']:
        return (gate, p_idx, (q_idx, None))
    elif gate in ['H', 'I']:
        return (gate, None, (q_idx, None))
    elif gate == 'CNOT':
        return (gate, None, (q_idx, t_idx))
    else:
        raise ValueError("Unknown gate")


class GPTQE(GPT):
    def calculate_loss(self, ops_tensor, energies):
        """
        ops_tensor: (B,T,4) : (g,q,p,t), p/t는 -1 가능
        energies:   (B,1)
        """
        B, T, _ = ops_tensor.shape
        device = ops_tensor.device

        # 1) BOS(0) 붙이고, 전체 시퀀스를 오른쪽으로 한 칸 밀어 넣는다 (:-1 쓰지 말기)
        bos = torch.zeros((B, 1), dtype=torch.long, device=device)
        q_in = torch.cat([bos, (ops_tensor[:, :, 1].clamp(min=0) + 1)], dim=1)  # (B, T+1)
        g_in = torch.cat([bos, (ops_tensor[:, :, 0].clamp(min=0) + 1)], dim=1)
        p_in = torch.cat([bos, (ops_tensor[:, :, 2].clamp(min=0) + 1)], dim=1)
        t_in = torch.cat([bos, (ops_tensor[:, :, 3].clamp(min=0) + 1)], dim=1)

        out = self(q_in, g_in, p_in, t_in)  # logits length = T+1
        # 2) 예측은 1:부터 T+1까지, 정답은 ops_tensor의 0:부터 T-1까지 => 둘 다 길이 T
        q_log = out["q"][:, 1:, :]  # (B,T,Q)
        g_log = out["g"][:, 1:, :]  # (B,T,G)
        p_log = out["p"][:, 1:, :]  # (B,T,P)
        t_log = out["t"][:, 1:, :]  # (B,T,Q)

        g_idx = ops_tensor[:, :, 0]  # (B,T)
        q_idx = ops_tensor[:, :, 1]
        p_idx = ops_tensor[:, :, 2]
        t_idx = ops_tensor[:, :, 3]

        # 3) gather로 안전하게 뽑기
        log_q = q_log.gather(2, q_idx.unsqueeze(-1)).squeeze(-1)  # (B,T)
        log_g = g_log.gather(2, g_idx.unsqueeze(-1)).squeeze(-1)

        log_p = torch.zeros_like(log_q)
        mask_p = p_idx >= 0
        if mask_p.any():
            log_p[mask_p] = p_log.gather(2, p_idx.clamp(min=0).unsqueeze(-1)).squeeze(-1)[mask_p]

        log_t = torch.zeros_like(log_q)
        mask_t = t_idx >= 0
        if mask_t.any():
            log_t[mask_t] = t_log.gather(2, t_idx.clamp(min=0).unsqueeze(-1)).squeeze(-1)[mask_t]

        total_logits = (log_q + log_g + log_p + log_t).sum(dim=1)  # (B,)
        loss = torch.mean((total_logits - energies.squeeze()) ** 2)
        return loss

    @torch.no_grad()
    def generate(self, n_sequences, max_new_tokens, temperature=1.0, device="cpu"):
        B = n_sequences
        q_ids = torch.zeros((B, 1), dtype=torch.long, device=device)
        g_ids = torch.zeros((B, 1), dtype=torch.long, device=device)
        p_ids = torch.zeros((B, 1), dtype=torch.long, device=device)
        t_ids = torch.zeros((B, 1), dtype=torch.long, device=device)

        ops = []
        total_energy_dummy = torch.zeros((B, 1), device=device)

        for _ in range(max_new_tokens):
            out = self(q_ids, g_ids, p_ids, t_ids)
            # last position logits
            q_log = out["q"][:, -1, :]
            g_log = out["g"][:, -1, :]
            p_log = out["p"][:, -1, :]
            t_log = out["t"][:, -1, :]

            # sample q
            q_prob = F.softmax(q_log / temperature, dim=-1)
            q_sample = torch.multinomial(q_prob, 1).squeeze(1)

            # sample g
            g_prob = F.softmax(g_log / temperature, dim=-1)
            g_sample = torch.multinomial(g_prob, 1).squeeze(1)

            # param?
            is_param_gate = g_sample.unsqueeze(1) == torch.tensor([0, 1, 2], device=device)  # RX,RY,RZ
            use_p = is_param_gate.any(dim=1)
            p_sample = torch.full_like(q_sample, -1)
            if use_p.any():
                p_prob = F.softmax(p_log[use_p] / temperature, dim=-1)
                p_sample[use_p] = torch.multinomial(p_prob, 1).squeeze(1)

            # target?
            is_cnot = g_sample == 3
            t_sample = torch.full_like(q_sample, -1)
            if is_cnot.any():
                t_logits_sel = t_log[is_cnot]
                t_mask = torch.ones_like(t_logits_sel, dtype=torch.bool)
                # control != target
                ctrl = q_sample[is_cnot]
                t_mask[torch.arange(t_mask.size(0)), ctrl] = False
                t_logits_sel = t_logits_sel.masked_fill(~t_mask, float('-inf'))
                t_prob = F.softmax(t_logits_sel / temperature, dim=-1)
                t_sample[is_cnot] = torch.multinomial(t_prob, 1).squeeze(1)

            ops.append(torch.stack([g_sample, q_sample, p_sample, t_sample], dim=1))

            # append for next conditioning (+1 shift, -1->0)
            q_ids = torch.cat([q_ids, (q_sample + 1).unsqueeze(1)], dim=1)
            g_ids = torch.cat([g_ids, (g_sample + 1).unsqueeze(1)], dim=1)
            p_ids = torch.cat([p_ids, (p_sample.clamp(min=0) + 1).unsqueeze(1)], dim=1)
            t_ids = torch.cat([t_ids, (t_sample.clamp(min=0) + 1).unsqueeze(1)], dim=1)

        ops = torch.stack(ops, dim=1)  # (B,T,4)
        return ops, total_energy_dummy


@qml.qnode(dev, interface='torch')
def fidelity_circuit(x1, x2, circuit):
    apply_circuit(x1, circuit)
    qml.adjoint(apply_circuit)(x2, circuit)
    return qml.probs(wires=range(num_qubit))


def compute_energy_for_ops(ops_and_data):
    ops_indices, X1, X2, Y = ops_and_data
    energy_per_seq = []
    for single_x1, single_x2, single_y in zip(X1, X2, Y):
        # convert indices to gate tuples
        circuit = [indices_to_gate_tuple(int(g), int(q), int(p), int(t))
                   for (g, q, p, t) in ops_indices]
        probs = fidelity_circuit(single_x1, single_x2, circuit)
        es = abs(probs[0] - single_y)
        energy_per_seq.append(es.item())
    return np.mean(energy_per_seq)


def get_sequence_energies(op_seq, X1, X2, Y, num_workers=1):
    with Pool(processes=num_workers) as pool:
        inputs = [(ops, X1, X2, Y) for ops in op_seq]
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
    max_epoch = 20
    gate_type = ['RX', 'RY', 'RZ', 'CNOT', 'H', 'I']
    op_pool = make_op_pool(gate_type=gate_type, num_qubit=num_qubit, num_param=num_qubit)
    op_pool_size = len(op_pool)
    train_size = 32
    n_batches = 8
    max_gate = 20
    T_max = 1000
    T_min = 0.01


    X_train, _, Y_train, _ = data_load_and_process("kmnist", reduction_sz=num_qubit, train_len=population_size)

    config = GPTConfig(
        block_size=max_gate + 1,  # <-- +1
        dropout=0.2,
        bias=False,
        num_qubits=num_qubit,
        num_gates=len(gate_type),
        num_params=num_qubit
    )
    gpt = GPTQE(config)
    opt = gpt.configure_optimizers(weight_decay=0.01, learning_rate=5e-5, betas=(0.9, 0.999), device_type="cpu")
    gpt.train()

    X1, X2, Y, data_store = new_data(batch_size, X_train, Y_train)
    mu, sigma = None, None

    fidelity_history = []
    loss_history = []
    all_gen_records = []

    for i in range(max_epoch):
        gpt.eval()
        train_ops_torch, _ = gpt.generate(
            n_sequences=train_size * 3,
            max_new_tokens=max_gate,
            temperature=temperature(T_max=T_max, T_min=T_min, max_epoch=max_epoch, epoch=i),
            device="cpu",
        )
        gpt.train()

        train_ops_np = train_ops_torch.cpu().numpy()  # (N,T,4)
        train_seq_en = get_sequence_energies(train_ops_np, X1, X2, Y)

        alpha = 0.1
        if mu is None:
            mu, sigma = float(train_seq_en.mean()), float(train_seq_en.std()) + 1e-8
        else:
            mu = alpha * float(train_seq_en.mean()) + (1 - alpha) * mu
            sigma = alpha * float(train_seq_en.std()) + (1 - alpha) * sigma
        print(f"[scale] μ={mu:.6f}, σ={sigma:.6f}")

        train_seq_en_norm = normalize_E(train_seq_en, mu, sigma)
        train_ops_np, train_seq_en_norm = select_token_and_en(train_ops_np, train_seq_en_norm, train_size)

        ops_tensor = torch.from_numpy(train_ops_np).long()
        energies = torch.from_numpy(train_seq_en_norm)

        token_batches = torch.tensor_split(ops_tensor, n_batches)
        energy_batches = torch.tensor_split(energies, n_batches)

        loss_record = 0
        for token_batch, energy_batch in zip(token_batches, energy_batches):
            opt.zero_grad()
            loss = gpt.calculate_loss(token_batch, energy_batch)
            loss.backward()
            opt.step()
            loss_record += loss.item() / n_batches
        loss_history.append(loss_record)

        # eval
        gpt.eval()
        gen_ops, _ = gpt.generate(
            n_sequences=100,
            max_new_tokens=max_gate,
            temperature=0.01,
            device="cpu"
        )
        gen_ops_np = gen_ops.numpy()
        true_Es = get_sequence_energies(gen_ops_np, X1, X2, Y)
        true_Es_norm = normalize_E(true_Es, mu, sigma)

        ave_E = np.mean(true_Es)
        print(f"Iter: {i + 1}, Loss: {loss_record}, Ave True E: {ave_E}")

        fidelity_history.append(ave_E)
        record_generated_results(all_gen_records, i + 1,
                                 [[indices_to_gate_tuple(int(g), int(q), int(p), int(t))
                                   for (g, q, p, t) in seq] for seq in gen_ops_np],
                                 true_Es)

    name = 'hier_factorized'

    plot_result(fidelity_history, f'{name}_fidelity', f'{name}_fidelity.png')
    plot_result(loss_history, f'{name}_loss', f'{name}_loss.png')
    with open(f"{name}_generated_circuit.json", "w") as f:
        json.dump(all_gen_records, f, indent=2)
    with open(f"{name}_data_store.pkl", "wb") as f:
        pickle.dump(data_store, f)
