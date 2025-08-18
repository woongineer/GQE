import json
import pickle
import random
import time
import os
from multiprocessing import Pool

import pandas as pd
import pennylane as qml
import plotly.graph_objects as go
import tensorflow as tf
import torch
from pennylane import numpy as pnp
from sklearn.decomposition import PCA
from torch import nn
from tqdm import tqdm

dev = qml.device('default.qubit', wires=4)


def get_circuit(filename):
    print('getting circuit...')
    with open(filename, 'r') as file:
        data = json.load(file)
    df = pd.DataFrame(data)
    df['aveTrueE'] = df.groupby('GeneratedIteration')['energy'].transform('mean')
    return df


def get_data(filename):
    print('getting data...')
    with open(filename, "rb") as f:
        df = pickle.load(f)
    raw_X, raw_Y, processed_data = df['raw_X'], df['raw_Y'], df['processed']
    return raw_X, raw_Y, processed_data


def get_circuit_by_energy(data, top_or_bottom, n_circuit):
    print(f'extracting {top_or_bottom}...')
    top_or_bottom = True if top_or_bottom == 'top' else False
    selected_df = data.sort_values(by='energy', ascending=top_or_bottom)[:1000]
    rm_duple = selected_df.drop_duplicates(subset=["gen_op_seq"])
    return rm_duple[:n_circuit]


def make_random_circuit(with_scale, gate_type, max_gate, num_qubits, scales):
    circuit = []
    for _ in range(max_gate):
        gate = random.choice(gate_type)

        if gate in ['H', 'I']:
            target = random.randint(0, num_qubits - 1)
            circuit.append([gate, None, [target, None]])

        elif gate == 'CNOT':
            control, target = random.sample(range(num_qubits), 2)
            circuit.append([gate, None, [control, target]])

        elif gate in ['RX', 'RY', 'RZ']:
            param_idx = random.randint(0, num_qubits - 1)
            target = random.randint(0, num_qubits - 1)
            if with_scale:
                scale = random.choice(scales)
                param_idx = [param_idx, scale]
            circuit.append([gate, param_idx, [target, None]])

        elif gate == 'MultiRZ':
            param_idx = random.randint(0, num_qubits - 1)
            control, target = random.sample(range(num_qubits), 2)
            if with_scale:
                scale = random.choice(scales)
                param_idx = [param_idx, scale]
            circuit.append([gate, param_idx, [control, target]])

    return circuit


def get_ZZEmbedding(N_layers):
    def exp_Z(x, wires):
        qml.RZ(-2 * x, wires=wires)

    def exp_ZZ2(x1, x2, wires):
        qml.CNOT(wires=wires)
        qml.RZ(-2 * (pnp.pi - x1) * (pnp.pi - x2), wires=wires[1])
        qml.CNOT(wires=wires)

    def ZZEmbedding(input):
        for i in range(N_layers):
            for j in range(4):
                qml.Hadamard(wires=j)
                exp_Z(input[j], wires=j)
            for k in range(3):
                exp_ZZ2(input[k], input[k + 1], wires=[k, k + 1])
            exp_ZZ2(input[3], input[0], wires=[3, 0])

    return ZZEmbedding


def build_circuit(gate_seq=None, with_scale=False, N_layers=None):
    ZZEmbedding = get_ZZEmbedding(N_layers) if gate_seq is None else None

    @qml.qnode(dev, interface="torch")
    def circuit(inputs):
        if gate_seq is None:
            ZZEmbedding(inputs[0:4])
            qml.adjoint(lambda: ZZEmbedding(inputs[4:8]))()
        else:
            apply_structure(gate_seq, with_scale, inputs[0:4])
            qml.adjoint(lambda: apply_structure(gate_seq, with_scale, inputs[4:8]))()
        return qml.probs(wires=range(4))

    return circuit


def apply_structure(gate_seq, with_scale, x):
    for gate_name, param, wires in gate_seq:
        if gate_name == "H":
            qml.Hadamard(wires=wires[0])
        elif gate_name == "I":
            qml.Identity(wires=wires[0])
        elif gate_name == "CNOT":
            qml.CNOT(wires=wires)  # [control, target]
        elif gate_name == "RX":
            theta = x[param[0]] * param[1] if with_scale else x[param]
            qml.RX(theta, wires=wires[0])
        elif gate_name == "RY":
            theta = x[param[0]] * param[1] if with_scale else x[param]
            qml.RY(theta, wires=wires[0])
        elif gate_name == "RZ":
            theta = x[param[0]] * param[1] if with_scale else x[param]
            qml.RZ(theta, wires=wires[0])
        elif gate_name == "MultiRZ":
            theta = x[param[0]] * param[1] if with_scale else x[param]
            qml.MultiRZ(theta, wires=wires)  # [w1, w2, ...]


class Model_Fidelity(torch.nn.Module):
    def __init__(self, qnode):
        super().__init__()
        self.qnode = qnode
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
        )

    def forward(self, x1, x2):
        x1 = self.linear_relu_stack(x1)
        x2 = self.linear_relu_stack(x2)
        x = torch.cat([x1, x2], dim=1)
        # 수동 loop
        return torch.stack([self.qnode(xi) for xi in x])[:, 0]


# make new data for hybrid model
def new_data(batch_size, X, Y):
    X1_new, X2_new, Y_new = [], [], []
    for i in range(batch_size):
        n, m = pnp.random.randint(len(X)), pnp.random.randint(len(X))
        X1_new.append(X[n])
        X2_new.append(X[m])
        if Y[n] == Y[m]:
            Y_new.append(1)
        else:
            Y_new.append(0)

    X1_new = torch.tensor(pnp.array(X1_new), dtype=torch.float32)
    X2_new = torch.tensor(pnp.array(X2_new), dtype=torch.float32)

    Y_new = torch.tensor(pnp.array(Y_new), dtype=torch.float64)

    return X1_new, X2_new, Y_new


def population_data(train_len=400):
    data_path = "/Users/jwheo/Desktop/Y/NQE/Neural-Quantum-Embedding/rl/kmnist"
    # data_path = "GQE/kmnist"
    kmnist_train_images_path = f"{data_path}/kmnist-train-imgs.npz"
    kmnist_train_labels_path = f"{data_path}/kmnist-train-labels.npz"

    x_train = pnp.load(kmnist_train_images_path)["arr_0"]
    y_train = pnp.load(kmnist_train_labels_path)["arr_0"]

    x_train = x_train[..., pnp.newaxis] / 255.0
    train_filter_tf = pnp.where((y_train == 0) | (y_train == 1))

    x_train, y_train = x_train[train_filter_tf], y_train[train_filter_tf]

    x_train = tf.image.resize(x_train[:], (256, 1)).numpy()
    x_train = tf.squeeze(x_train).numpy()

    X_train = PCA(4).fit_transform(x_train)
    x_train = []
    for x in X_train:
        x = (x - x.min()) * (2 * pnp.pi / (x.max() - x.min()))
        x_train.append(x)
    return x_train[:train_len], y_train[:train_len]


def run_NQE_compare(data_x, data_y, with_scale, N_layer, batch_size, epoch, good_circuits, bad_circuits, rand_circuits, ave_len):
    print("Running NQE comparison...")
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    models = {}

    for i in range(len(good_circuits)):
        gate_seq = good_circuits.iloc[i]['gen_op_seq']
        circuit = build_circuit(gate_seq=gate_seq, with_scale=with_scale)
        models[f"G{i + 1}"] = Model_Fidelity(circuit)

    for i in range(len(bad_circuits)):
        gate_seq = bad_circuits.iloc[i]['gen_op_seq']
        circuit = build_circuit(gate_seq=gate_seq, with_scale=with_scale)
        models[f"B{i + 1}"] = Model_Fidelity(circuit)

    for i in range(len(rand_circuits)):
        circuit = build_circuit(gate_seq=rand_circuits[i], with_scale=with_scale)
        models[f"R{i + 1}"] = Model_Fidelity(circuit)

    models["zz"] = Model_Fidelity(build_circuit(N_layers=N_layer))

    opts = {name: torch.optim.SGD(model.parameters(), lr=0.01) for name, model in models.items()}
    loss_lists = {name: [] for name in models.keys()}
    loss_fn = torch.nn.MSELoss()

    for it in range(epoch):
        X1_batch, X2_batch, Y_batch = new_data(batch_size, data_x, data_y)
        # print(f"Epoch {it + 1}/{epoch}...")
        for name, model in models.items():
            pred = model(X1_batch, X2_batch)
            loss = loss_fn(pred, Y_batch)
            opts[name].zero_grad()
            loss.backward()
            opts[name].step()
            loss_lists[name].append(loss.item())

    final_energy = {name: pnp.mean(energy[-ave_len:]) for name, energy in loss_lists.items()}

    return final_energy


def run_NQE_compare_wrapper(args):
    """Added to ALTER seed in multiprocessing for server."""
    seed = int(time.time() * 1e6) % (2**32 - 1) + os.getpid()
    random.seed(seed)
    pnp.random.seed(seed)
    torch.manual_seed(seed)

    print(f"[Worker PID {os.getpid()}] Using seed: {seed}")
    return run_NQE_compare(**args)


def run_multiple_NQE_compare(n_repeat, num_workers, **kwargs):
    task_args = [kwargs.copy() for _ in range(n_repeat)]

    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(run_NQE_compare_wrapper, task_args),
            total=n_repeat,
            desc="Running multiple NQE_analysis_and_circuit_draw experiments"
        ))
    return results


def get_color(key):
    if key.startswith("G"):
        return "#0D28BF"
    elif key.startswith("B"):
        return "red"
    elif key.startswith("R"):
        return "#1487B5"
    elif key == "zz":
        return "black"
    else:
        return "gray"


def plot_energy_errorbars(energy_list, html_path="energy_errorbar_plot.html", width=2000, height=600, filter_keys=None):
    df = pd.DataFrame(energy_list)

    if filter_keys is not None:
        df = df[filter_keys]

    mean_vals = df.mean()
    std_vals = df.std()

    fig = go.Figure()

    for key in df.columns:
        color = get_color(key)
        fig.add_trace(go.Scatter(
            x=[key],
            y=[mean_vals[key]],
            mode='markers',
            marker=dict(color=color, size=6),
            error_y=dict(type='data', array=[std_vals[key]], color=color, thickness=1.5),
            name=key,
            showlegend=True
        ))

    fig.update_layout(
        title="NQEed Energy per Circuit (Mean ± Std)",
        xaxis_title="Circuit",
        yaxis_title="Energy",
        xaxis_tickangle=-90,
        width=width,
        height=height,
        template="plotly_white",
        showlegend=True,
    )

    fig.write_html(html_path)
    print(f"graph save: {html_path}")


if __name__ == "__main__":
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    circuit_filename = 'fix_sample_SM_more_gate_generated_circuit.json'
    data_filename = 'fix_sample_SM_more_gate_data_store.pkl'
    with_scale = True

    n_circuit = 3
    batch_size = 25
    N_layer = 1
    epoch = 50
    averaging_length = 10
    num_cpus = 2
    repeat = 2
    plot_width = 1000

    gate_type = ['RX', 'RY', 'RZ', 'CNOT', 'H', 'I']
    max_gate = 20
    num_qubits = 4
    scales = [0.1, 0.3, 0.5, 0.7, 1]

    circuits = get_circuit(circuit_filename)
    data_x, data_y, _ = get_data(data_filename)
    # original_x, original_y = population_data(train_len=400)

    good_circuits = get_circuit_by_energy(circuits, 'top', n_circuit=n_circuit)
    bad_circuits = get_circuit_by_energy(circuits, 'bottom', n_circuit=n_circuit)

    rand_circuits = [make_random_circuit(with_scale, gate_type, max_gate, num_qubits, scales) for _ in range(n_circuit)]

    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    energy_list = run_multiple_NQE_compare(n_repeat=repeat,
                                           num_workers=num_cpus,
                                           data_x=data_x,
                                           data_y=data_y,
                                           with_scale=with_scale,
                                           N_layer=N_layer,
                                           batch_size=batch_size,
                                           epoch=epoch,
                                           good_circuits=good_circuits,
                                           bad_circuits=bad_circuits,
                                           rand_circuits=rand_circuits,
                                           ave_len=averaging_length)

    plot_energy_errorbars(energy_list, html_path="e222nergy_errorbar_epoch_200.html", width=plot_width)
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
