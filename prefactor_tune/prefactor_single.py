import json
import pickle

import pandas as pd
import pennylane as qml
import torch
from pennylane import numpy as pnp
from torch import nn, optim


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
    Y_new  = torch.tensor(pnp.array(Y_new),  dtype=torch.float32)  # MSELoss와 맞춤

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


def count_param_gates(gate_seq):
    return sum(g[1] is not None for g in gate_seq)


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


def build_qnode_with_bias(num_qubits, gate_seq):
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev, interface="torch")
    def qnode(packed):
        # packed = [x1(4), x2(4), bias(num_bias)]
        x1 = packed[:4]
        x2 = packed[4:8]
        bias = packed[8:]

        apply_structure(gate_seq, x1, bias)
        qml.adjoint(lambda: apply_structure(gate_seq, x2, bias))()
        return qml.probs(wires=range(num_qubits))

    return qnode


if __name__ == "__main__":
    circuit_filename = 'fix_sample_SM_more_gate_generated_circuit.json'
    data_filename = 'fix_sample_SM_more_gate_data_store.pkl'

    batch_size = 25
    epoch = 7

    num_qubits = 4

    circuits = get_circuit(circuit_filename)
    data_x, data_y, _ = get_data(data_filename)

    selected_circuit = circuits.sort_values(by='energy', ascending=True).iloc[:3]
    test_circuit = selected_circuit.iloc[2]
    gate_seq = test_circuit.gen_op_seq

    num_of_bias = count_param_gates(gate_seq)

    bias_net = BiasNet(num_of_bias=num_of_bias, num_of_feature=num_qubits)
    bias_opt = optim.SGD(bias_net.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    qnode = build_qnode_with_bias(num_qubits, gate_seq)

    for it in range(epoch):
        bias_opt.zero_grad()

        X1_batch, X2_batch, Y_batch = new_data(batch_size, data_x, data_y)
        bias = bias_net(X1_batch, X2_batch)

        preds = []
        for i in range(batch_size):
            packed = torch.cat([X1_batch[i], X2_batch[i], bias[i]], dim=0)
            probs = qnode(packed)
            preds.append(probs[0])

        preds = torch.stack(preds).float()
        loss = loss_fn(preds, Y_batch)
        loss.backward()
        bias_opt.step()

