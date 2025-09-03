import json
import logging
import os
import pickle
import random
import time
from multiprocessing import get_context

import pandas as pd
import pennylane as qml
import plotly.graph_objects as go
import torch
from numpy import pi
from pennylane import numpy as pnp
from torch import nn, optim

logger = logging.getLogger("prefactor")


########################## util functions ##########################

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


########################## Graph ##########################


def get_color(key):
    if key.startswith("G"):
        return "#0D28BF"
    elif key.startswith("B"):
        return "red"
    elif key.startswith("R"):
        return "#1487B5"
    elif key == "zz":
        return "black"
    elif key == "zz_nqe":
        return "#2E7D32"
    else:
        return "gray"


def plot_energy_errorbars(energy_list, html_path="energy_errorbar.html", filter_keys=None):
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
        template="plotly_white",
        showlegend=True,
    )

    fig.write_html(html_path)
    logger.info(f"graph save: {html_path}")


def plot_epoch_trajectories(trace_dict, html_path="epoch_trajectories.html",
                            title="Energy vs Epoch (per model)", filter_keys=None):
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


def plot_initial_final_arrows(trace_dict, html_path="init_final_arrows.html",
                              ave_len=1, filter_keys=None,
                              title="Init → Final Energy (per model)"):
    fig = go.Figure()
    keys = list(trace_dict.keys()) if filter_keys is None else filter_keys
    xs = list(range(len(keys)))

    fig.update_xaxes(tickmode="array", tickvals=xs, ticktext=keys)

    for i, name in enumerate(keys):
        ys = trace_dict[name]
        if len(ys) == 0:
            continue

        y0 = float(ys[0])
        L = max(1, min(ave_len, len(ys)))
        y1 = float(pnp.mean(ys[-L:]))

        base = get_color(name)

        fig.add_trace(go.Scatter(
            x=[xs[i]], y=[y0], mode="markers",
            marker=dict(symbol="x", size=7, line=dict(width=1, color=base), color=base),
            name=f"{name} (init)"
        ))
        fig.add_trace(go.Scatter(
            x=[xs[i]], y=[y1], mode="markers",
            marker=dict(symbol="circle", size=5, color=base),
            name=f"{name} (final)"
        ))

        fig.add_annotation(
            x=xs[i], y=y1, ax=xs[i], ay=y0,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowwidth=1.5,
            arrowcolor=base, opacity=0.6
        )

    fig.update_layout(
        title=title,
        xaxis_title="Model",
        yaxis_title="Energy",
        template="plotly_white",
        showlegend=True,
    )
    fig.write_html(html_path)
    logger.info(f"graph save: {html_path}")


########################## Runner ##########################

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


def run_compare(data_x, data_y, num_qubits, n_layer, batch_size, epoch, good_circuits, bad_circuits, rand_circuits,
                ave_len):
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
        opts[name] = optim.SGD(nets[name].parameters(), lr=0.001)
        qnodes[name] = build_qnode_with_bias(num_qubits, gate_seq)

    for i in range(len(bad_circuits)):
        gate_seq = bad_circuits.iloc[i]['gen_op_seq']
        num_of_bias = count_param_gates(gate_seq)
        name = f"B{i + 1}"

        nets[name] = BiasNet(num_of_bias=num_of_bias, num_of_feature=num_qubits)
        opts[name] = optim.SGD(nets[name].parameters(), lr=0.001)
        qnodes[name] = build_qnode_with_bias(num_qubits, gate_seq)

    for i in range(len(rand_circuits)):
        gate_seq = rand_circuits[i]
        num_of_bias = count_param_gates(gate_seq)
        name = f"R{i + 1}"

        nets[name] = BiasNet(num_of_bias=num_of_bias, num_of_feature=num_qubits)
        opts[name] = optim.SGD(nets[name].parameters(), lr=0.001)
        qnodes[name] = build_qnode_with_bias(num_qubits, gate_seq)

    zz_bias_num = count_param_gates_zz(n_layer)
    nets["zz"] = BiasNet(num_of_bias=zz_bias_num, num_of_feature=num_qubits)
    opts["zz"] = optim.SGD(nets["zz"].parameters(), lr=0.001)
    qnodes["zz"] = build_zz_qnode_with_bias(num_qubits, n_layer)

    nets["zz_nqe"] = EncoderNet(num_of_feature=num_qubits)
    opts["zz_nqe"] = optim.SGD(nets["zz_nqe"].parameters(), lr=0.001)
    qnodes["zz_nqe"] = build_zz_qnode_nqe(num_qubits, n_layer)

    loss_lists = {name: [] for name in nets.keys()}
    loss_fn = torch.nn.MSELoss()

    for it in range(epoch):
        X1_batch, X2_batch, Y_batch = new_data(batch_size, data_x, data_y)
        logger.info(f"Epoch {it + 1}/{epoch}...")
        for name, model in nets.items():
            opts[name].zero_grad()
            if name == "zz_nqe":
                packed8 = nets[name](X1_batch, X2_batch)
                preds = []
                for i in range(batch_size):
                    probs = qnodes[name](packed8[i])
                    preds.append(probs[0])
                preds = torch.stack(preds).float()
            else:
                bias = nets[name](X1_batch, X2_batch)
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


########################## Bias Network ##########################

class BiasNet(nn.Module):
    def __init__(self, num_of_bias, num_of_feature):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Linear(num_of_feature, num_of_feature * 4), nn.ReLU(),
            nn.Linear(num_of_feature * 4, num_of_feature * 2), nn.ReLU(),
        )
        self.head = nn.Linear(num_of_feature * 2, num_of_bias, bias=True)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x1, x2):
        x1 = self.feat(x1)
        x2 = self.feat(x2)
        b1 = self.head(x1)
        b2 = self.head(x2)
        return torch.cat([b1, b2], dim=1)


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
        # packed = [x1(4), x2(4), b1(L), b2(L)]
        x1 = packed[:4]
        x2 = packed[4:8]
        bcat = packed[8:]
        L = bcat.shape[0] // 2
        b1 = bcat[:L]
        b2 = bcat[L:]

        apply_structure(gate_seq, x1, b1)
        qml.adjoint(lambda: apply_structure(gate_seq, x2, b2))()
        return qml.probs(wires=range(num_qubits))

    return qnode


#########################for zz circuit#########################
def count_param_gates_zz(n_layers):
    return 8 * n_layers


def apply_zz_with_bias(n_layers, x, bias):
    bias_count = 0
    for _ in range(n_layers):
        for j in range(4):
            qml.Hadamard(wires=j)
            theta = -2.0 * x[j] + bias[bias_count]
            bias_count += 1
            qml.RZ(theta, wires=j)

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
        bcat = packed[8:]
        L = bcat.shape[0] // 2
        b1 = bcat[:L]
        b2 = bcat[L:]

        apply_zz_with_bias(n_layers, x1, b1)
        qml.adjoint(lambda: apply_zz_with_bias(n_layers, x2, b2))()
        return qml.probs(wires=range(num_qubits))

    return qnode


class EncoderNet(nn.Module):
    def __init__(self, num_of_feature):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(num_of_feature, num_of_feature * 2), nn.ReLU(),
            nn.Linear(num_of_feature * 2, num_of_feature * 2), nn.ReLU(),
            nn.Linear(num_of_feature * 2, num_of_feature),
        )

    def forward(self, x1, x2):
        z1 = self.enc(x1)
        z2 = self.enc(x2)
        return torch.cat([z1, z2], dim=1)


def apply_zz_no_bias(n_layers, x):
    for _ in range(n_layers):
        for j in range(4):
            qml.Hadamard(wires=j)
            qml.RZ(-2.0 * x[j], wires=j)
        for k in range(3):
            qml.CNOT(wires=[k, k + 1])
            qml.RZ(-2.0 * ((pi - x[k]) * (pi - x[k + 1])), wires=k + 1)
            qml.CNOT(wires=[k, k + 1])
        qml.CNOT(wires=[3, 0])
        qml.RZ(-2.0 * ((pi - x[3]) * (pi - x[0])), wires=0)
        qml.CNOT(wires=[3, 0])


def build_zz_qnode_nqe(num_qubits, n_layers):
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev, interface="torch")
    def qnode(packed8):
        # packed8 = [z1(4), z2(4)] ; bias 없음
        z1 = packed8[:4]
        z2 = packed8[4:8]
        apply_zz_no_bias(n_layers, z1)
        qml.adjoint(lambda: apply_zz_no_bias(n_layers, z2))()
        return qml.probs(wires=range(num_qubits))

    return qnode


if __name__ == "__main__":
    logger = setup_logger()
    logger.info("Starting prefactor analysis...")

    # os.environ.setdefault("OMP_NUM_THREADS", "1")
    # os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    # os.environ.setdefault("MKL_NUM_THREADS", "1")
    # os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    # os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    # os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
    # os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
    # torch.set_num_threads(1)

    # circuit_filename = 'prefactor_tune/28_main_more_gate_generated_circuit.json'
    # data_filename = 'prefactor_tune/28_main_more_gate_data_store.pkl'
    circuit_filename = '28_main_more_gate_generated_circuit.json'
    data_filename = '28_main_more_gate_data_store.pkl'

    batch_size = 25
    n_layer = 1
    n_circuit = 3  # 50
    epoch = 10  # 100
    averaging_length = 2  # 10
    num_cpus = 2  # 16
    repeat = num_cpus
    # html_filename = f"prefactor_tune/indept_add_epoch{epoch}"
    html_filename = f"epoch{epoch}"

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
    plot_energy_errorbars(energy_list, html_path=f"{html_filename}_errorbar.html")

    trace_repeat_idx = 0
    plot_epoch_trajectories(results[trace_repeat_idx]["trace"],
                            html_path=f"{html_filename}_trajectory.html", title=f"Energy vs Epoch")
    plot_initial_final_arrows(results[trace_repeat_idx]["trace"], html_path=f"{html_filename}_arrow.html",
                              ave_len=averaging_length, title=f"Init(X) → Final(O) Energy")
