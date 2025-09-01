import json
import pickle
import logging
import pandas as pd
import os
import time
import pennylane as qml
import random
from numpy import pi
import torch
from pennylane import numpy as pnp
from multiprocessing import get_context
import plotly.graph_objects as go
from torch import nn, optim

logger = logging.getLogger("prefactor")


def setup_logger():
    global logger
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s | %(process)d | %(levelname)s | %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def new_data(batch_size, X, Y):
    X1_new, X2_new, Y_new = [], [], []
    for _ in range(batch_size):
        n, m = pnp.random.randint(len(X)), pnp.random.randint(len(X))
        X1_new.append(X[n])
        X2_new.append(X[m])
        if Y[n] == Y[m]:
            Y_new.append(1)
        else:
            Y_new.append(0)

    X1_new = torch.tensor(pnp.array(X1_new), dtype=torch.float32)
    X2_new = torch.tensor(pnp.array(X2_new), dtype=torch.float32)
    Y_new = torch.tensor(pnp.array(Y_new), dtype=torch.float32)

    return X1_new, X2_new, Y_new


def get_circuit(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    df = pd.DataFrame(data)
    df['aveTrueE'] = df.groupby('GeneratedIteration')['energy'].transform('mean')
    return df


def get_data(filename):
    with open(filename, "rb") as f:
        df = pickle.load(f)
    raw_X, raw_Y, processed_data = df['raw_X'], df['raw_Y'], df['processed']
    return raw_X, raw_Y, processed_data


def get_circuit_by_energy(data, top_or_bottom, n_circuit):
    logger.info(f'extracting {top_or_bottom}...')
    top_or_bottom = top_or_bottom == 'top'
    selected_df = data.sort_values(by='energy', ascending=top_or_bottom)[:1000]
    rm_duple = selected_df.drop_duplicates(subset=["gen_op_seq"])
    return rm_duple[:n_circuit]


def count_param_gates(gate_seq):
    return sum(g[1] is not None for g in gate_seq)


def make_random_circuit(gate_type, max_gate, num_qubits, scales):
    circuit = []
    for _ in range(max_gate):
        gate = random.choice(gate_type)

        if gate in ['H', 'I']:
            target = random.randint(0, num_qubits - 1)
            circuit.append([gate, None, [target, None]])

        elif gate == 'CNOT':
            control, target = random.sample(range(num_qubits), 2)
            circuit.append([gate, None, [control, target]])

        elif gate in ['RX', 'RY', 'RZ', 'MultiRZ']:
            param_idx = random.randint(0, num_qubits - 1)
            control, target = random.sample(range(num_qubits), 2)
            scale = random.choice(scales)
            param_idx = [param_idx, scale]
            if gate == 'MultiRZ':
                circuit.append([gate, param_idx, [control, target]])
            else:
                circuit.append([gate, param_idx, [target, None]])

    return circuit


class BiasNet(nn.Module):
    def __init__(self, num_of_bias, num_of_feature):
        super().__init__()
        self.x_stack = nn.Sequential(
            nn.Linear(num_of_feature, num_of_feature * 2), nn.ReLU(),
            nn.Linear(num_of_feature * 2, num_of_feature * 2), nn.ReLU(),
            nn.Linear(num_of_feature * 2, num_of_feature),
        )
        self.bias_head = nn.Sequential(
            nn.Linear(num_of_feature * 2, num_of_feature * 4), nn.ReLU(),
            nn.Linear(num_of_feature * 4, num_of_bias),
        )

    def forward(self, x1, x2):
        x1 = self.x_stack(x1)
        x2 = self.x_stack(x2)
        x = torch.cat([x1, x2], dim=1)
        return self.bias_head(x)


def apply_structure(gate_seq, x, bias):
    bias_count = 0
    for gate_name, param, wires in gate_seq:
        if gate_name == "H":
            qml.Hadamard(wires=wires[0])
        elif gate_name == "I":
            qml.Identity(wires=wires[0])
        elif gate_name == "CNOT":
            qml.CNOT(wires=wires)
        elif gate_name in ["RX", "RY", "RZ", "MultiRZ"]:
            theta = x[param[0]] * param[1] + bias[bias_count]
            bias_count += 1
            if gate_name == "RX":
                qml.RX(theta, wires=wires[0])
            elif gate_name == "RY":
                qml.RY(theta, wires=wires[0])
            elif gate_name == "RZ":
                qml.RZ(theta, wires=wires[0])
            elif gate_name == "MultiRZ":
                qml.MultiRZ(theta, wires=wires)
    assert bias_count == len(bias), "bias length mismatch"


def build_qnode_with_bias(num_qubits, gate_seq):
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev, interface="torch")
    def qnode(packed):
        # packed = [x1(4), x2(4), bias(num_bias)]
        x1 = packed[:4]
        x2 = packed[4:8]
        bias = packed[8:]

        apply_structure(gate_seq, x1, bias)
        qml.adjoint(lambda: apply_structure(gate_seq, x2, -bias))()
        return qml.probs(wires=range(num_qubits))

    return qnode


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
        title="Prefactored Energy per Circuit (Mean ± Std)",
        xaxis_title="Circuit",
        yaxis_title="Energy",
        xaxis_tickangle=-90,
        # width=width,
        # height=height,
        template="plotly_white",
        showlegend=True,
    )

    fig.write_html(html_path)
    logger.info(f"graph save: {html_path}")


def plot_epoch_trajectories(trace_dict, html_path="epoch_trajectories.html",
                            title="Energy vs Epoch (per model)", filter_keys=None):
    """
    trace_dict: {model_name: [energy_at_epoch1, energy_at_epoch2, ...]}
    """
    fig = go.Figure()
    keys = list(trace_dict.keys()) if filter_keys is None else filter_keys
    for name in keys:
        ys = trace_dict[name]
        xs = list(range(1, len(ys) + 1))
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=name))
    fig.update_layout(
        title=title,
        xaxis_title="Epoch",
        yaxis_title="Energy",
        template="plotly_white",
        showlegend=True,
    )
    fig.write_html(html_path)
    logger.info(f"trajectory graph save: {html_path}")


def run_multiple_compare(n_repeat, num_workers, **kwargs):
    task_args = [kwargs.copy() for _ in range(n_repeat)]

    ctx = get_context("spawn")
    with ctx.Pool(processes=num_workers, maxtasksperchild=1) as pool:
        results = list(pool.imap(run_compare_wrapper, task_args, chunksize=1))
    return results


def run_compare_wrapper(args):
    logger = setup_logger()
    torch.set_num_threads(1)
    seed = int(time.time() * 1e6) % (2 ** 32 - 1) + os.getpid()
    random.seed(seed)
    pnp.random.seed(seed)
    torch.manual_seed(seed)

    logger.info(f"[Worker PID {os.getpid()}] Using seed: {seed}")
    return run_compare(**args)


def run_compare(data_x, data_y, num_qubits, n_layer, batch_size, epoch, good_circuits, bad_circuits, rand_circuits, ave_len):
    logger.info("Running comparison...")
    logger.info(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    opts = {}
    nets = {}
    qnodes = {}
    for i in range(len(good_circuits)):
        gate_seq = good_circuits.iloc[i]['gen_op_seq']
        num_of_bias = count_param_gates(gate_seq)
        name = f"G{i + 1}"

        nets[name] = BiasNet(num_of_bias=num_of_bias, num_of_feature=num_qubits)
        opts[name] = optim.SGD(nets[name].parameters(), lr=0.01)
        qnodes[name] = build_qnode_with_bias(num_qubits, gate_seq)

    for i in range(len(bad_circuits)):
        gate_seq = bad_circuits.iloc[i]['gen_op_seq']
        num_of_bias = count_param_gates(gate_seq)
        name = f"B{i + 1}"

        nets[name] = BiasNet(num_of_bias=num_of_bias, num_of_feature=num_qubits)
        opts[name] = optim.SGD(nets[name].parameters(), lr=0.01)
        qnodes[name] = build_qnode_with_bias(num_qubits, gate_seq)

    for i in range(len(rand_circuits)):
        gate_seq = rand_circuits[i]
        num_of_bias = count_param_gates(gate_seq)
        name = f"R{i + 1}"

        nets[name] = BiasNet(num_of_bias=num_of_bias, num_of_feature=num_qubits)
        opts[name] = optim.SGD(nets[name].parameters(), lr=0.01)
        qnodes[name] = build_qnode_with_bias(num_qubits, gate_seq)

    zz_bias_num = count_param_gates_zz(n_layer)
    nets["zz"] = BiasNet(num_of_bias=zz_bias_num, num_of_feature=num_qubits)
    opts["zz"] = optim.SGD(nets["zz"].parameters(), lr=0.01)
    qnodes["zz"] = build_zz_qnode_with_bias(num_qubits, n_layer)

    loss_lists = {name: [] for name in nets.keys()}
    loss_fn = torch.nn.MSELoss()

    for it in range(epoch):
        X1_batch, X2_batch, Y_batch = new_data(batch_size, data_x, data_y)
        logger.info(f"Epoch {it + 1}/{epoch}...")
        for name, model in nets.items():
            opts[name].zero_grad()
            bias = nets[name](X1_batch, X2_batch)
            bias = 0.2 * torch.tanh(bias) #수정부

            preds = []
            for i in range(batch_size):
                packed = torch.cat([X1_batch[i], X2_batch[i], bias[i]], dim=0)
                probs = qnodes[name](packed)
                preds.append(probs[0])

            preds = torch.stack(preds).float()
            loss = loss_fn(preds, Y_batch)
            loss.backward()
            opts[name].step()
            loss_lists[name].append(loss.item())

    final_energy = {name: pnp.mean(energy[-ave_len:]) for name, energy in loss_lists.items()}

    return {"final": final_energy, "trace": loss_lists}


#########################for zz circuit#########################
def count_param_gates_zz(n_layers):
    return 8 * n_layers


def apply_zz_with_bias(n_layers, x, bias):
    bias_count = 0
    for _ in range(n_layers):
        # Hadamard + local Z rotations
        for j in range(4):
            qml.Hadamard(wires=j)
            theta = -2.0 * x[j] + bias[bias_count]
            bias_count += 1
            qml.RZ(theta, wires=j)

        # ZZ couplings: (0,1), (1,2), (2,3), (3,0)
        for k in range(3):
            qml.CNOT(wires=[k, k + 1])
            theta = -2.0 * ((pi - x[k]) * (pi - x[k + 1])) + bias[bias_count]
            bias_count += 1
            qml.RZ(theta, wires=k + 1)
            qml.CNOT(wires=[k, k + 1])

        qml.CNOT(wires=[3, 0])
        theta = -2.0 * ((pi - x[3]) * (pi - x[0])) + bias[bias_count]
        bias_count += 1
        qml.RZ(theta, wires=0)
        qml.CNOT(wires=[3, 0])

    assert bias_count == len(bias), "zz bias length mismatch"


def build_zz_qnode_with_bias(num_qubits, n_layers):
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev, interface="torch")
    def qnode(packed):
        x1 = packed[:4]
        x2 = packed[4:8]
        bias = packed[8:]

        apply_zz_with_bias(n_layers, x1, bias)
        qml.adjoint(lambda: apply_zz_with_bias(n_layers, x2, -bias))()
        return qml.probs(wires=range(num_qubits))

    return qnode


if __name__ == "__main__":
    logger = setup_logger()
    logger.info("Starting prefactor analysis...")

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
    torch.set_num_threads(1)

    circuit_filename = 'prefactor_analysis/28_main_more_gate_generated_circuit.json'
    data_filename = 'prefactor_analysis/28_main_more_gate_data_store.pkl'

    n_circuit = 50
    batch_size = 25
    n_layer = 1
    epoch = 100
    averaging_length = 10
    num_cpus = 16
    repeat = 16
    html_filename = f"prefactor_analysis/errorbar_epoch_{epoch}.html"

    gate_type = ['RX', 'RY', 'RZ', 'CNOT', 'H', 'I']
    max_gate = 28
    num_qubits = 4
    scales = [0.1, 0.3, 0.5, 0.7, 1]

    circuits = get_circuit(circuit_filename)
    data_x, data_y, _ = get_data(data_filename)

    good_circuits = get_circuit_by_energy(circuits, 'top', n_circuit=n_circuit)
    bad_circuits = get_circuit_by_energy(circuits, 'bottom', n_circuit=n_circuit)
    rand_circuits = [make_random_circuit(gate_type, max_gate, num_qubits, scales) for _ in range(n_circuit)]

    logger.info('Analysis begin...')

    results = run_multiple_compare(n_repeat=repeat,
                                   num_workers=num_cpus,
                                   data_x=data_x,
                                   data_y=data_y,
                                   num_qubits=num_qubits,
                                   n_layer=n_layer,
                                   batch_size=batch_size,
                                   epoch=epoch,
                                   good_circuits=good_circuits,
                                   bad_circuits=bad_circuits,
                                   rand_circuits=rand_circuits,
                                   ave_len=averaging_length)

    logger.info('Analysis finished...')

    energy_list = [res["final"] for res in results]
    plot_energy_errorbars(energy_list, html_path=html_filename)

    trace_repeat_idx = 0
    plot_epoch_trajectories(results[trace_repeat_idx]["trace"],
                            html_path=f"prefactor_analysis/epoch_traj_repeat_{trace_repeat_idx}.html",
                            title=f"Energy vs Epoch (repeat {trace_repeat_idx})")
