import pennylane as qml
import tensorflow as tf
import torch
import pickle
from pennylane import numpy as np
from sklearn.decomposition import PCA
import numpy as onp
import matplotlib.pyplot as plt
from torch import nn

dev = qml.device('default.qubit', wires=4)


###########수정된 부분##########
def plot_loss_curve(loss_history, title="Training Loss per Epoch", savepath="NQE_ref.png", show=False):
    """
    loss_history: 각 epoch/iteration에서의 loss 값 list[float]
    savepath: 저장 파일 경로 (기본 'NQE_ref.png')
    show: True면 plt.show() 시도(에러 시 자동으로 save로 폴백)
    """
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(range(1, len(loss_history) + 1), loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(title)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    if show:
        try:
            plt.show()
        except Exception as e:
            print(f"[matplotlib] show() 실패, 파일 저장으로 폴백: {e}")
            plt.savefig(savepath, bbox_inches="tight", dpi=150)
    else:
        plt.savefig(savepath, bbox_inches="tight", dpi=150)

    plt.close()
###########수정된 부분##########



def get_data(filename):
    with open(filename, "rb") as f:
        df = pickle.load(f)
    raw_X, raw_Y, processed_data = df['raw_X'], df['raw_Y'], df['processed']
    return raw_X, raw_Y, processed_data


# exp(ixZ) gate
def exp_Z(x, wires):
    qml.RZ(-2 * x, wires=wires)


# exp(i(pi - x1)(pi - x2)ZZ) gate
def exp_ZZ2(x1, x2, wires):
    qml.CNOT(wires=wires)
    qml.RZ(-2 * (np.pi - x1) * (np.pi - x2), wires=wires[1])
    qml.CNOT(wires=wires)


# Quantum Embedding 1 for model 1 (Conventional ZZ feature embedding)
def QuantumEmbedding(x):
    # x: (..., 4)  마지막 축이 feature, 앞쪽은 배치 축(없어도 됨)
    for i in range(N_layers):
        for j in range(4):
            qml.Hadamard(wires=j)
            exp_Z(x[..., j], wires=j)  # 배치 브로드캐스팅
        for k in range(3):
            exp_ZZ2(x[..., k], x[..., k + 1], wires=[k, k + 1])  # 배치 브로드캐스팅
        exp_ZZ2(x[..., 3], x[..., 0], wires=[3, 0])


@qml.qnode(dev, interface="torch")
def circuit(inputs):
    # inputs: (B, 8) 또는 (8,)
    QuantumEmbedding(inputs[..., 0:4])            # 마지막 축 기준 슬라이싱
    qml.adjoint(QuantumEmbedding)(inputs[..., 4:8])
    return qml.probs(wires=range(4))


class Model_Fidelity(torch.nn.Module):
    def __init__(self):
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
    X1_new, X2_new, Y_new = torch.tensor(X1_new).to(
        torch.float32), torch.tensor(X2_new).to(torch.float32), torch.tensor(
        Y_new).to(torch.float32)
    return X1_new, X2_new, Y_new




if __name__ == "__main__":
    # load data
    data_filename = '../prefactor_tune/28_main_more_gate_data_store.pkl'
    X_train, Y_train, _ = get_data(data_filename)

    N_layers = 1

    batch_size = 25
    iterations = 1000

    model = Model_Fidelity()
    model.train()

    loss_fn = torch.nn.MSELoss()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_history = []
    for it in range(iterations):
        X1_batch, X2_batch, Y_batch = new_data(batch_size, X_train, Y_train)
        pred = model(X1_batch, X2_batch)
        loss = loss_fn(pred, Y_batch)

        opt.zero_grad()
        loss.backward()
        opt.step()

        loss_history.append(loss.item())

        print(f"Iterations: {it} Loss: {loss.item()}")

    plot_loss_curve(loss_history, title="Training Loss per Epoch", savepath="NQE_ref.png", show=False)
