import ast
import torch
import pandas as pd
import pennylane as qml
from torch import nn
from NOTUSE.my_legacy_data_fix_py_is_first_hope.data import data_load_and_process, new_data

dev = qml.device('default.qubit', wires=4)

# ============================ 회로 생성 함수 ============================

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

def make_circuit(gate_seq):
    @qml.qnode(dev, interface="torch")
    def circuit(inputs):
        apply_structure(gate_seq, inputs[0:4])
        qml.adjoint(lambda: apply_structure(gate_seq, inputs[4:8]))()
        return qml.probs(wires=range(4))
    return circuit

# ============================ ZZ 회로 정의 ============================

def ZZ(input):
    for j in range(4):
        qml.Hadamard(wires=j)
        qml.RZ(-1 * input[j], wires=j)
    for k in range(3):
        qml.CNOT(wires=[k, k+1])
        qml.RZ(-1 * (qml.numpy.pi - input[k]) * (qml.numpy.pi - input[k + 1]), wires=k+1)
        qml.CNOT(wires=[k, k+1])
    qml.CNOT(wires=[3, 0])
    qml.RZ(-1 * (qml.numpy.pi - input[3]) * (qml.numpy.pi - input[0]), wires=0)
    qml.CNOT(wires=[3, 0])

@qml.qnode(dev, interface="torch")
def zz_circuit(inputs):
    ZZ(inputs[0:4])
    qml.adjoint(ZZ)(inputs[4:8])
    return qml.probs(wires=range(4))

# ============================ 모델 정의 ============================

class Model_Fidelity(nn.Module):
    def __init__(self, circuit):
        super().__init__()
        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes={})
        self.encoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 4)
        )

    def forward(self, x1, x2):
        x1 = self.encoder(x1)
        x2 = self.encoder(x2)
        x = torch.concat([x1, x2], dim=1)
        return self.qlayer(x)[:, 0]

# ============================ 메인 학습 루프 ============================

if __name__ == "__main__":
    df = pd.read_csv('selected_circ.csv')
    df = df.drop_duplicates(subset=["gen_op_seq"])

    # 데이터 로드
    X_train, _, Y_train, _ = data_load_and_process("kmnist", reduction_sz=4, train_len=400)

    # 하이퍼파라미터
    batch_size = 25
    iterations = 10
    N_models = 3  # 사용할 회로 수 (원하면 100으로 늘려도 됨)

    # 모델 및 옵티마이저 생성
    circuits = [zz_circuit] + [make_circuit(ast.literal_eval(df.iloc[i]['gen_op_seq'])) for i in range(N_models)]
    models = [Model_Fidelity(qml.qnn.TorchLayer(c, weight_shapes={})) for c in circuits]
    opts = [torch.optim.SGD(m.parameters(), lr=0.01) for m in models]
    loss_fn = nn.MSELoss()
    loss_records = [[] for _ in range(N_models + 1)]

    for it in range(iterations):
        X1_batch, X2_batch, Y_batch = new_data(batch_size, X_train, Y_train)
        for i, model in enumerate(models):
            model.train()
            pred = model(X1_batch, X2_batch)
            loss = loss_fn(pred, Y_batch)
            opts[i].zero_grad()
            loss.backward()
            opts[i].step()
            loss_records[i].append(loss.item())

        if it % 2 == 0:
            print(f"Iter {it}: ", [round(l[-1], 4) for l in loss_records])
