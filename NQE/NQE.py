import pickle

import pennylane as qml
import ast
import torch
import pandas as pd
from pennylane import numpy as np
import plotly.graph_objs as go
from torch import nn

dev = qml.device('default.qubit', wires=4)

def data_load():
    with open("data_fix_sampling_SM_data_store.pkl", "rb") as f:
        data = pickle.load(f)
    return data


def exp_Z(x, wires):
    qml.RZ(-2 * x, wires=wires)


# exp(i(pi - x1)(pi - x2)ZZ) gate
def exp_ZZ2(x1, x2, wires):
    qml.CNOT(wires=wires)
    qml.RZ(-2 * (np.pi - x1) * (np.pi - x2), wires=wires[1])
    qml.CNOT(wires=wires)


# Quantum Embedding 1 for model 1 (Conventional ZZ feature embedding)
def ZZEmbedding(input):
    for i in range(N_layers):
        for j in range(4):
            qml.Hadamard(wires=j)
            exp_Z(input[j], wires=j)
        for k in range(3):
            exp_ZZ2(input[k], input[k + 1], wires=[k, k + 1])
        exp_ZZ2(input[3], input[0], wires=[3, 0])


def build_circuit(gate_seq=None):
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
        n, m = np.random.randint(len(X)), np.random.randint(len(X))
        X1_new.append(X[n])
        X2_new.append(X[m])
        if Y[n] == Y[m]:
            Y_new.append(1)
        else:
            Y_new.append(0)

    X1_new = torch.tensor(np.array(X1_new), dtype=torch.float32)
    X2_new = torch.tensor(np.array(X2_new), dtype=torch.float32)

    Y_new = torch.tensor(np.array(Y_new), dtype=torch.float64)


    return X1_new, X2_new, Y_new


def plot_loss_lists(loss_lists, output_path="model_loss_plot.html"):
    fig = go.Figure()
    for model_name, losses in loss_lists.items():
        fig.add_trace(go.Scatter(
            y=losses,
            mode="lines+markers",
            name=model_name
        ))

    fig.update_layout(
        title="Loss Over Iterations per Model",
        xaxis_title="Iteration",
        yaxis_title="Loss",
        legend_title="Model",
        template="plotly_white"
    )

    fig.write_html(output_path)
    print(f"✅ Plot saved to {output_path}")



if __name__ == "__main__":
    # load data
    loaded_data = data_load()
    raw_X, raw_Y, processed_data = loaded_data['raw_X'], loaded_data['raw_Y'], loaded_data['processed']

    df = pd.read_csv('selected_circ.csv')
    df = df.drop_duplicates(subset=["gen_op_seq"])

    N_layers = 2
    batch_size = 25
    iterations = 300
    n_models = 20

    models = {}
    for i in range(n_models):
        gate_seq = ast.literal_eval(df.iloc[i]['gen_op_seq'])
        circuit = build_circuit(gate_seq)
        models[f"model_{i}"] = Model_Fidelity(circuit)
    models["zz"] = Model_Fidelity(build_circuit())

    opts = {name: torch.optim.SGD(model.parameters(), lr=0.01) for name, model in models.items()}
    loss_lists = {name: [] for name in models.keys()}
    loss_fn = torch.nn.MSELoss()

    for it in range(iterations):
        X1_batch, X2_batch, Y_batch = new_data(batch_size, raw_X, raw_Y)
        for name, model in models.items():
            pred = model(X1_batch, X2_batch)
            loss = loss_fn(pred, Y_batch)
            opts[name].zero_grad()
            loss.backward()
            opts[name].step()
            loss_lists[name].append(loss.item())

        print(f"Iterations: {it} Loss: {loss.item()}")

    plot_loss_lists(loss_lists, "model_loss_plot_2.html")
