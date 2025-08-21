import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml


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


def select_token_and_en(train_token_seq, train_seq_en, train_size):
    k = int(train_size * 0.4)
    middle = train_size - (k * 2)

    sorted_indices = np.argsort(train_seq_en[:, -1])

    top_indices = sorted_indices[:k]
    bottom_indices = sorted_indices[-k:]

    middle_pool_indices = sorted_indices[k:-k]
    interval_points = np.linspace(0, len(middle_pool_indices) - 1, num=middle)
    middle_sample_indices_in_pool = np.round(interval_points).astype(int)
    middle_indices = middle_pool_indices[middle_sample_indices_in_pool]

    final_indices = np.concatenate([top_indices, bottom_indices, middle_indices])
    np.random.shuffle(final_indices)

    new_train_token_seq = train_token_seq[final_indices]
    new_train_seq_en = train_seq_en[final_indices]

    return new_train_token_seq, new_train_seq_en, final_indices


def plot_result(data, title, filename):
    plt.figure(figsize=(10, 5))
    plt.plot(data, marker='o')
    plt.title(title)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def record_generated_results(all_records, iteration, gen_op_seq, energies):
    for op, e in zip(gen_op_seq, energies):
        # 각 gate tuple을 list로 바꿔 JSON 직렬화 가능하게
        op_serialized = [[g if not isinstance(g, np.ndarray) else g.tolist() for g in gate] for gate in op]
        record = {
            "GeneratedIteration": iteration,
            "gen_op_seq": op_serialized,
            "energy": float(e)
        }
        all_records.append(record)
