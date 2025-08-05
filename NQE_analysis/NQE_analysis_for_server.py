"""NQE Analysis code for server environment.
This code use multithreading property fit to the server specification.
"""
import json
import logging
import os
import pickle
import random
import time
from concurrent.futures import ThreadPoolExecutor
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


def setup_logger():
    logger = logging.getLogger()
    if not logger.hasHandlers():
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(process)d | %(levelname)s | %(message)s',
            handlers=[logging.StreamHandler()]
        )
    return logger


def get_circuit(filename):
    logger.info('getting circuit...')
    with open(filename, 'r') as file:
        data = json.load(file)
    df = pd.DataFrame(data)
    df['aveTrueE'] = df.groupby('GeneratedIteration')['energy'].transform('mean')
    return df


def get_data(filename):
    logger.info('getting data...')
    with open(filename, "rb") as f:
        df = pickle.load(f)
    raw_X, raw_Y, processed_data = df['raw_X'], df['raw_Y'], df['processed']
    return raw_X, raw_Y, processed_data


def get_circuit_by_energy(data, top_or_bottom, n_circuit):
    logger.info(f'extracting {top_or_bottom}...')
    top_or_bottom = True if top_or_bottom == 'top' else False
    selected_df = data.sort_values(by='energy', ascending=top_or_bottom)[:1000]
    rm_duple = selected_df.drop_duplicates(subset=["gen_op_seq"])
    return rm_duple[:n_circuit]


def make_random_circuit(gate_type, max_gate, num_qubits):
    circuit = []
    for _ in range(max_gate):
        gate = random.choice(gate_type)

        if gate in ['RX', 'RY', 'RZ']:
            param_idx = random.randint(0, 3)  # 예: 0~3 사이 임의 파라미터 index
            target = random.randint(0, num_qubits - 1)
            circuit.append([gate, param_idx, [target, None]])

        elif gate == 'CNOT':
            control, target = random.sample(range(num_qubits), 2)  # 서로 다른 두 qubit
            circuit.append([gate, None, [control, target]])

        elif gate in ['H', 'I']:
            target = random.randint(0, num_qubits - 1)
            circuit.append([gate, None, [target, None]])

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


def build_circuit(gate_seq=None, N_layers=None):
    ZZEmbedding = get_ZZEmbedding(N_layers) if gate_seq is None else None

    @qml.qnode(dev, interface="torch")
    def circuit(inputs):
        if gate_seq is None:
            ZZEmbedding(inputs[0:4])
            qml.adjoint(lambda: ZZEmbedding(inputs[4:8]))()
        else:
            apply_structure(gate_seq, inputs[0:4])
            qml.adjoint(lambda: apply_structure(gate_seq, inputs[4:8]))()
        return qml.probs(wires=range(4))

    return circuit


def apply_structure(gate_seq, x):
    for gate_name, target, param_indices in gate_seq:
        if gate_name == "CNOT":
            qml.CNOT(wires=param_indices)
        else:
            param_idx = param_indices[0]
            angle = x[param_idx] if param_idx is not None else 0.0
            if gate_name == "RX":
                qml.RX(angle, wires=target)
            elif gate_name == "RY":
                qml.RY(angle, wires=target)
            elif gate_name == "RZ":
                qml.RZ(angle, wires=target)


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
    data_path = "GQE/kmnist"
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


def train_single_model(name, model, X1, X2, Y, opt, loss_fn):
    pred = model(X1, X2)
    loss = loss_fn(pred, Y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    return name, loss.item()


def run_NQE_compare(data_x, data_y, N_layer, batch_size, epoch, good_circuits, bad_circuits, rand_circuits, ave_len):
    models = {}

    for i in range(len(good_circuits)):
        gate_seq = good_circuits.iloc[i]['gen_op_seq']
        circuit = build_circuit(gate_seq)
        models[f"G{i + 1}"] = Model_Fidelity(circuit)

    for i in range(len(bad_circuits)):
        gate_seq = bad_circuits.iloc[i]['gen_op_seq']
        circuit = build_circuit(gate_seq)
        models[f"B{i + 1}"] = Model_Fidelity(circuit)

    for i in range(len(rand_circuits)):
        circuit = build_circuit(rand_circuits[i])
        models[f"R{i + 1}"] = Model_Fidelity(circuit)

    models["zz"] = Model_Fidelity(build_circuit(N_layers=N_layer))

    opts = {name: torch.optim.SGD(model.parameters(), lr=0.01) for name, model in models.items()}
    loss_lists = {name: [] for name in models.keys()}
    loss_fn = torch.nn.MSELoss()

    for it in range(epoch):
        X1_batch, X2_batch, Y_batch = new_data(batch_size, data_x, data_y)
        logger.info(f"Epoch {it + 1}/{epoch}...")

        with ThreadPoolExecutor(max_workers=2) as executor:  # 적절히 조절
            futures = [
                (name,
                 executor.submit(train_single_model, name, model, X1_batch, X2_batch, Y_batch, opts[name], loss_fn))
                for name, model in models.items()
            ]

            for name, future in futures:
                loss_val = future.result()
                loss_lists[name].append(loss_val)

    final_energy = {name: pnp.mean(energy[-ave_len:]) for name, energy in loss_lists.items()}

    return final_energy


def run_NQE_compare_wrapper(args):
    seed = int(time.time() * 1e6) % (2 ** 32 - 1) + os.getpid()
    random.seed(seed)
    pnp.random.seed(seed)
    torch.manual_seed(seed)

    logger.info(f"[Worker PID {os.getpid()}] Using seed: {seed}")
    return run_NQE_compare(**args)


def run_multiple_NQE_compare(n_repeat, num_workers, **kwargs):
    task_args = [kwargs.copy() for _ in range(n_repeat)]

    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(run_NQE_compare_wrapper, task_args),
            total=n_repeat,
            desc="Running multiple NQE experiments"
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


def plot_energy_errorbars(energy_list, html_path="energy_errorbar.html", width=2000, height=600, filter_keys=None):
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
    logger.info(f"graph save: {html_path}")


if __name__ == "__main__":
    logger = setup_logger()
    logger.info("Starting NQE Analysis...")
    circuit_filename = 'NQE_analysis/data_fix_sampling_SM_generated_circuit.json'
    data_filename = 'NQE_analysis/data_fix_sampling_SM_data_store.pkl'
    n_circuit = 50

    batch_size = 25
    N_layer = 1
    epoch = 100
    averaging_length = 10
    num_cpus = 16
    repeat = 16
    html_filename = f"NQE_analysis/errorbar_epoch_{epoch}_N_layer_{N_layer}.html"

    circuits = get_circuit(circuit_filename)
    data_x, data_y, _ = get_data(data_filename)
    # original_x, original_y = population_data(train_len=400)

    good_circuits = get_circuit_by_energy(circuits, 'top', n_circuit=n_circuit)
    bad_circuits = get_circuit_by_energy(circuits, 'bottom', n_circuit=n_circuit)

    gate_type = ['RX', 'RY', 'RZ', 'CNOT', 'H', 'I']
    max_gate = 20
    num_qubits = 4

    rand_circuits = [make_random_circuit(gate_type, max_gate, num_qubits) for _ in range(n_circuit)]

    logger.info('NQE begin...')

    energy_list = run_multiple_NQE_compare(n_repeat=repeat,
                                           num_workers=num_cpus,
                                           data_x=data_x,
                                           data_y=data_y,
                                           N_layer=N_layer,
                                           batch_size=batch_size,
                                           epoch=epoch,
                                           good_circuits=good_circuits,
                                           bad_circuits=bad_circuits,
                                           rand_circuits=rand_circuits,
                                           ave_len=averaging_length)

    logger.info('NQE finished...')

    plot_energy_errorbars(energy_list, html_path=html_filename)
