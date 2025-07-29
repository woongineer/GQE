import pennylane as qml
import torch
from pennylane import numpy as np
from torch import nn
import pandas as pd
from NOTUSE.my_legacy_data_fix_py_is_first_hope.data import data_load_and_process, new_data

dev = qml.device('default.qubit', wires=4)


# exp(ixZ) gate
def exp_Z(x, wires):
    qml.RZ(-1 * x, wires=wires)


# exp(i(pi - x1)(pi - x2)ZZ) gate
def exp_ZZ2(x1, x2, wires):
    qml.CNOT(wires=wires)
    qml.RZ(-1 * (np.pi - x1) * (np.pi - x2), wires=wires[1])
    qml.CNOT(wires=wires)


# Quantum Embedding 1 for model 1 (Conventional ZZ feature embedding)
def ZZ(input):
    for i in range(N_layers):
        for j in range(4):
            qml.Hadamard(wires=j)
            exp_Z(input[j], wires=j)
        for k in range(3):
            exp_ZZ2(input[k], input[k + 1], wires=[k, k + 1])
        exp_ZZ2(input[3], input[0], wires=[3, 0])

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

def my_circuit(gate_seq):
    @qml.qnode(dev, interface="torch")
    def circuit(inputs):
        apply_structure(gate_seq, inputs[0:4])
        qml.adjoint(lambda: apply_structure(gate_seq, inputs[4:8]))()
        return qml.probs(wires=range(4))
    return circuit


@qml.qnode(dev, interface="torch")
def zz_circuit(inputs):
    ZZ(inputs[0:4])
    qml.adjoint(ZZ)(inputs[4:8])
    return qml.probs(wires=range(4))


class Model_Fidelity(torch.nn.Module):
    def __init__(self, circuit):
        super().__init__()
        self.qlayer1 = qml.qnn.TorchLayer(circuit, weight_shapes={})
        self.linear_relu_stack1 = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 4)
        )

    def forward(self, x1, x2):
        x1 = self.linear_relu_stack1(x1)
        x2 = self.linear_relu_stack1(x2)
        x = torch.concat([x1, x2], 1)
        x = self.qlayer1(x)
        """you can use 
        fig, ax = qml.draw_mpl(circuit)(x)
        fig.savefig('dd.png')
        to see the circuit
        """
        return x[:, 0]


if __name__ == "__main__":
    df = pd.read_csv('selected_circ.csv')
    df = df.drop_duplicates(subset=["gen_op_seq"])
    # load data
    X_train, _, Y_train, _ = data_load_and_process("kmnist", reduction_sz=4, train_len=400)

    N_layers = 3

    batch_size = 25
    iterations = 7

    model = Model_Fidelity(zz_circuit)
    model.train()

    loss_fn = torch.nn.MSELoss()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    for it in range(iterations):
        X1_batch, X2_batch, Y_batch, _ = new_data(batch_size, X_train, Y_train)
        pred = model(X1_batch, X2_batch)
        loss = loss_fn(pred, Y_batch)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if it % 3 == 0:
            print(f"Iterations: {it} Loss: {loss.item()}")

    torch.save(model.state_dict(), "model.pt")

