import os

import numpy as np
import numpy as pnp
import pennylane as qml
import tensorflow as tf
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.svm import SVC


def data_load_and_process(reduction_sz: int = 4, train_len=400, test_len=100,
                          train_start: int = 0, test_start: int = 0):
    data_path = "/Users/jwheo/Desktop/Y/NQE/Neural-Quantum-Embedding/rl/kmnist"

    kmnist_train_images_path = f"{data_path}/kmnist-train-imgs.npz"
    kmnist_train_labels_path = f"{data_path}/kmnist-train-labels.npz"
    kmnist_test_images_path = f"{data_path}/kmnist-test-imgs.npz"
    kmnist_test_labels_path = f"{data_path}/kmnist-test-labels.npz"

    x_train = pnp.load(kmnist_train_images_path)["arr_0"]
    y_train = pnp.load(kmnist_train_labels_path)["arr_0"]
    x_test = pnp.load(kmnist_test_images_path)["arr_0"]
    y_test = pnp.load(kmnist_test_labels_path)["arr_0"]

    x_train, x_test = (x_train[..., pnp.newaxis] / 255.0, x_test[..., pnp.newaxis] / 255.0)
    train_filter = pnp.where((y_train == 0) | (y_train == 1))
    test_filter = pnp.where((y_test == 0) | (y_test == 1))
    x_train, y_train = x_train[train_filter], y_train[train_filter]
    x_test, y_test = x_test[test_filter], y_test[test_filter]

    x_train = tf.image.resize(x_train, (256, 1)).numpy()
    x_test = tf.image.resize(x_test, (256, 1)).numpy()
    x_train, x_test = tf.squeeze(x_train).numpy(), tf.squeeze(x_test).numpy()

    x_tr_sel = x_train[train_start:train_start + train_len]
    y_tr_sel = y_train[train_start:train_start + train_len]
    x_te_sel = x_test[test_start:test_start + test_len]
    y_te_sel = y_test[test_start:test_start + test_len]

    pca = PCA(reduction_sz).fit(x_tr_sel)
    X_train = pca.transform(x_tr_sel)
    X_test = pca.transform(x_te_sel)

    x_train_out, x_test_out = [], []
    for x in X_train:
        rng = (x.max() - x.min())
        x = (x - x.min()) * (2 * pnp.pi / (rng if rng != 0 else 1.0))
        x_train_out.append(x)
    for x in X_test:
        rng = (x.max() - x.min())
        x = (x - x.min()) * (2 * pnp.pi / (rng if rng != 0 else 1.0))
        x_test_out.append(x)

    return x_train_out, x_test_out, y_tr_sel, y_te_sel



def zz_feature_map(x, wires):
    n = len(wires)
    for i in range(n):
        qml.Hadamard(wires=wires[i])
        qml.RZ(x[i], wires=wires[i])
    for i in range(n):
        for j in range(i + 1, n):
            qml.CNOT(wires=[wires[i], wires[j]])
            qml.RZ(2.0 * x[i] * x[j], wires=wires[j])
            qml.CNOT(wires=[wires[i], wires[j]])


# 자리표시자(필요시 대체). 실제 user 임베딩은 아래 ckpt 로더로 생성해서 씀.
def _placeholder_user_embedding(x, wires):
    for i in range(len(wires)):
        qml.RY(x[i], wires=wires[i])


# =========================
# 2) 오버랩 커널 팩토리
# =========================
def make_overlap_kernel(embedding_fn, num_qubits: int):
    dev = qml.device("default.qubit", wires=num_qubits, shots=None)

    @qml.qnode(dev)
    def overlap_circuit(x1, x2):
        wires = list(range(num_qubits))
        embedding_fn(x1, wires=wires)
        qml.adjoint(embedding_fn)(x2, wires=wires)
        return qml.probs(wires=wires)

    def kernel_fn(x1, x2):
        return overlap_circuit(x1, x2)[0]

    return kernel_fn


# =========================
# 3) SVM 학습/평가
# =========================
def fit_eval_svm(X_train, y_train, X_test, y_test, kernel_callable, C: float = 0.1):
    K_train = qml.kernels.kernel_matrix(X_train, X_train, kernel_callable)
    clf = SVC(C=C, kernel="precomputed").fit(K_train, y_train)

    train_acc = clf.score(K_train, y_train)

    K_test = qml.kernels.kernel_matrix(X_test, X_train, kernel_callable)
    test_acc = clf.score(K_test, y_test)
    return train_acc, test_acc


# =========================
# 4) BiasNet 로더 → user_embedding 생성
# =========================
def load_bias_model(ckpt_path):
    payload = torch.load(ckpt_path, map_location="cpu")
    return payload["state_dict"], payload["arch_info"]  # arch_info: {'type': 'structured', 'gate_seq':..., ...}


def count_param_gates(gate_seq):
    return sum(g[1] is not None for g in gate_seq)


class _FeatHead(nn.Module):
    def __init__(self, num_of_bias, num_of_feature):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Linear(num_of_feature, num_of_feature * 4), nn.ReLU(),
            nn.Linear(num_of_feature * 4, num_of_feature * 2), nn.ReLU(),
        )
        self.head = nn.Linear(num_of_feature * 2, num_of_bias, bias=True)

    def forward(self, x):
        z = self.feat(x)
        return self.head(z)


def build_single_input_bias_net(state_dict: dict, num_bias: int, num_feat: int):
    net = _FeatHead(num_of_bias=num_bias, num_of_feature=num_feat)
    # state_dict의 키가 feat.*, head.*라고 가정. 다르면 아래 매핑에서 KeyError 날 수 있음.
    with torch.no_grad():
        for k, v in net.state_dict().items():
            if k in state_dict:
                v.copy_(state_dict[k])
            else:
                raise KeyError(f"state_dict key '{k}' not found in checkpoint.")
    net.eval()
    return net


def apply_structure(gate_seq, x, bias):
    """gate_seq: [ [gate_name, param(tuple or None), wires(list or tuple)], ... ]"""
    bidx = 0
    for gate_name, param, wires in gate_seq:
        if gate_name == "H":
            qml.Hadamard(wires=wires[0])
        elif gate_name == "I":
            qml.Identity(wires=wires[0])
        elif gate_name == "CNOT":
            qml.CNOT(wires=wires)
        elif gate_name in ["RX", "RY", "RZ", "MultiRZ"]:
            theta = x[param[0]] * param[1] + bias[bidx]
            bidx += 1
            if gate_name == "RX":
                qml.RX(theta, wires=wires[0])
            elif gate_name == "RY":
                qml.RY(theta, wires=wires[0])
            elif gate_name == "RZ":
                qml.RZ(theta, wires=wires[0])
            elif gate_name == "MultiRZ":
                qml.MultiRZ(theta, wires=wires)
    assert bidx == len(bias), f"bias length mismatch: used {bidx}, provided {len(bias)}"


def make_user_embedding_from_ckpt(ckpt_path: str, num_feat: int):
    state, arch = load_bias_model(ckpt_path)
    assert arch["type"] == "structured", "zz/zz_nqe가 아닌 'structured' 체크포인트만 사용하세요."
    gate_seq = arch["gate_seq"]
    num_bias = count_param_gates(gate_seq)
    net = build_single_input_bias_net(state, num_bias=num_bias, num_feat=num_feat)

    def user_embedding(x, wires):
        with torch.no_grad():
            xt = torch.tensor(x, dtype=torch.float32).view(1, -1)
            b = net(xt)[0].cpu().numpy()
        apply_structure(gate_seq, x, b)

    return user_embedding


# =========================
# 5) 메인
# =========================


class _EncoderNet(nn.Module):
    """네가 학습에 쓴 EncoderNet과 동일한 구조 (enc: 2* -> 2* -> out)"""
    def __init__(self, num_of_feature):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(num_of_feature, num_of_feature * 2), nn.ReLU(),
            nn.Linear(num_of_feature * 2, num_of_feature * 2), nn.ReLU(),
            nn.Linear(num_of_feature * 2, num_of_feature),
        )

    def forward(self, x):
        return self.enc(x)

def build_encoder_net(state_dict: dict, num_feat: int):
    net = _EncoderNet(num_of_feature=num_feat)
    with torch.no_grad():
        for k, v in net.state_dict().items():
            if k in state_dict:
                v.copy_(state_dict[k])
            else:
                raise KeyError(f"[zz_nqe] state_dict key '{k}' not found in checkpoint.")
    net.eval()
    return net

def apply_zz_no_bias(n_layers: int, z, wires):
    """학습 시 썼던 'bias 없음' ZZ 맵을 일반화 (임의 qubit 수, 링 커플링)"""
    n = len(wires)
    for _ in range(n_layers):
        # single-qubit part
        for i in range(n):
            qml.Hadamard(wires=wires[i])
            qml.RZ(-2.0 * z[i], wires=wires[i])
        # ring ZZ
        for i in range(n):
            j = (i + 1) % n
            qml.CNOT(wires=[wires[i], wires[j]])
            qml.RZ(-2.0 * ((np.pi - z[i]) * (np.pi - z[j])), wires=wires[j])
            qml.CNOT(wires=[wires[i], wires[j]])

def make_zz_nqe_embedding_from_ckpt(ckpt_path: str, num_feat: int):
    state, arch = load_bias_model(ckpt_path)
    assert arch["type"] == "zz_nqe", "이 체크포인트는 zz_nqe 타입이어야 합니다."
    n_layers = int(arch.get("n_layer", 1))

    enc = build_encoder_net(state, num_feat=num_feat)

    def zz_nqe_embedding(x, wires):
        with torch.no_grad():
            xt = torch.tensor(x, dtype=torch.float32).view(1, -1)
            z = enc(xt)[0].cpu().numpy()
        apply_zz_no_bias(n_layers, z, wires)

    return zz_nqe_embedding
if __name__ == "__main__":
    num_qubits = 4
    C = 0.1
    CKPT_PATH = "epoch1000_models/worker_pid3417964_20250906_045352/G1.pt"
    CKPT_PATH_ZZNQE = "epoch1000_models/worker_pid3417964_20250906_045352/zz_nqe.pt"

    X_train, X_test, y_train, y_test = data_load_and_process(reduction_sz=num_qubits,
                                                             train_len=300, test_len=50,
                                                             train_start=5000, test_start=1000)

    user_emb = make_user_embedding_from_ckpt(CKPT_PATH, num_feat=num_qubits)
    kernel_user = make_overlap_kernel(user_emb, num_qubits=num_qubits)
    kernel_zz = make_overlap_kernel(zz_feature_map, num_qubits=num_qubits)

    zznqe_emb = make_zz_nqe_embedding_from_ckpt(CKPT_PATH_ZZNQE, num_feat=num_qubits)
    kernel_zznqe = make_overlap_kernel(zz_nqe_embedding := zznqe_emb, num_qubits=num_qubits)

    za, zb = fit_eval_svm(X_train, y_train, X_test, y_test, kernel_zz, C=C)
    print(f"[ZZ-EMB  ] SVM accuracy: train={za:.3f}, test={zb:.3f}")

    ua, ub = fit_eval_svm(X_train, y_train, X_test, y_test, kernel_user, C=C)
    print(f"[USER-EMB] SVM accuracy: train={ua:.3f}, test={ub:.3f}")

    na, nb = fit_eval_svm(X_train, y_train, X_test, y_test, kernel_zznqe, C=C)
    print(f"[ZZ-NQE  ] SVM accuracy: train={na:.3f}, test={nb:.3f}")

    best = [("ZZ", zb), ("USER", ub), ("ZZ-NQE", nb)]
    best.sort(key=lambda t: t[1], reverse=True)
    print("=> 테스트 성능 순위:", " > ".join([f"{k}({v:.3f})" for k, v in best]))