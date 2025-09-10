import multiprocessing as mp
import os
from functools import partial

import numpy as np
import numpy as pnp
import pennylane as qml
import plotly.graph_objects as go
import tensorflow as tf
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.svm import SVC


def _set_single_thread_env():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
    try:
        torch.set_num_threads(1)
    except Exception:
        pass


def _worker_eval_ckpt(ckpt_file, ckpt_dir, num_qubits, C, X_train, X_test, y_train, y_test):
    _set_single_thread_env()

    ckpt_path = os.path.join(ckpt_dir, ckpt_file)
    if os.path.basename(ckpt_path) == "zz.pt":
        return None

    try:
        state, arch = load_bias_model(ckpt_path)
        arch_type = arch.get("type", "structured")
        label = None
        if arch_type == "structured":
            user_emb = make_user_embedding_from_ckpt(ckpt_path, num_feat=num_qubits)
            kernel = make_overlap_kernel(user_emb, num_qubits=num_qubits)
            label = os.path.splitext(os.path.basename(ckpt_path))[0]  # ex) G1, B3 ...
        elif arch_type == "zz_nqe":
            zznqe_emb = make_zz_nqe_embedding_from_ckpt(ckpt_path, num_feat=num_qubits)
            kernel = make_overlap_kernel(zznqe_emb, num_qubits=num_qubits)
            label = "zz_nqe"
        else:
            return None

        train_acc, test_acc = fit_eval_svm(X_train, y_train, X_test, y_test, kernel, C=C)
        return (ckpt_file, label, float(train_acc), float(test_acc))
    except Exception as e:
        return (ckpt_file, "ERROR", -1.0, -1.0, str(e))


def run_parallel_ckpts(ckpt_dir, num_workers, num_qubits, C, X_train, X_test, y_train, y_test):
    files = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt") and f != "zz.pt"]
    files.sort()

    worker = partial(
        _worker_eval_ckpt,
        ckpt_dir=ckpt_dir,
        num_qubits=num_qubits, C=C,
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
    )

    results = []
    with mp.Pool(processes=num_workers, maxtasksperchild=1) as pool:
        for out in pool.imap(worker, files, chunksize=1):
            if out is None:
                continue
            results.append(out)

    ok = [r for r in results if isinstance(r, tuple) and len(r) >= 4 and r[1] != "ERROR"]
    err = [r for r in results if isinstance(r, tuple) and len(r) >= 5 and r[1] == "ERROR"]

    ok.sort(key=lambda t: t[3], reverse=True)
    return ok, err


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


def fit_eval_svm(X_train, y_train, X_test, y_test, kernel_callable, C: float = 0.1):
    K_train = qml.kernels.kernel_matrix(X_train, X_train, kernel_callable)
    clf = SVC(C=C, kernel="precomputed").fit(K_train, y_train)

    train_acc = clf.score(K_train, y_train)

    K_test = qml.kernels.kernel_matrix(X_test, X_train, kernel_callable)
    test_acc = clf.score(K_test, y_test)
    return train_acc, test_acc


def load_bias_model(ckpt_path):
    payload = torch.load(ckpt_path, map_location="cpu")
    return payload["state_dict"], payload["arch_info"]


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
    with torch.no_grad():
        for k, v in net.state_dict().items():
            if k in state_dict:
                v.copy_(state_dict[k])
            else:
                raise KeyError(f"state_dict key '{k}' not found in checkpoint.")
    net.eval()
    return net


def apply_structure(gate_seq, x, bias):
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
    gate_seq = arch["gate_seq"]
    num_bias = count_param_gates(gate_seq)
    net = build_single_input_bias_net(state, num_bias=num_bias, num_feat=num_feat)

    def user_embedding(x, wires):
        with torch.no_grad():
            xt = torch.tensor(x, dtype=torch.float32).view(1, -1)
            b = net(xt)[0].cpu().numpy()
        apply_structure(gate_seq, x, b)

    return user_embedding


class _EncoderNet(nn.Module):
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
    n = len(wires)
    for _ in range(n_layers):
        for i in range(n):
            qml.Hadamard(wires=wires[i])
            qml.RZ(-2.0 * z[i], wires=wires[i])
        for i in range(n):
            j = (i + 1) % n
            qml.CNOT(wires=[wires[i], wires[j]])
            qml.RZ(-2.0 * ((np.pi - z[i]) * (np.pi - z[j])), wires=wires[j])
            qml.CNOT(wires=[wires[i], wires[j]])


def make_zz_nqe_embedding_from_ckpt(ckpt_path: str, num_feat: int):
    state, arch = load_bias_model(ckpt_path)
    n_layers = int(arch.get("n_layer", 1))

    enc = build_encoder_net(state, num_feat=num_feat)

    def zz_nqe_embedding(x, wires):
        with torch.no_grad():
            xt = torch.tensor(x, dtype=torch.float32).view(1, -1)
            z = enc(xt)[0].cpu().numpy()
        apply_zz_no_bias(n_layers, z, wires)

    return zz_nqe_embedding


def plot_bar_accuracies(ok_results, C, train_len, test_len, save_path="svm_results.html"):
    ok_sorted = sorted(ok_results, key=lambda t: t[3], reverse=True)
    labels     = [t[1] for t in ok_sorted]
    train_accs = [t[2] for t in ok_sorted]
    test_accs  = [t[3] for t in ok_sorted]

    fig = go.Figure()

    # Train bars
    fig.add_trace(go.Bar(
        x=labels,
        y=train_accs,
        name="Train",
        hovertemplate="Model: %{x}<br>Train acc: %{y:.6f}<extra></extra>",
    ))

    # Test bars
    fig.add_trace(go.Bar(
        x=labels,
        y=test_accs,
        name="Test",
        hovertemplate="Model: %{x}<br>Test acc: %{y:.6f}<extra></extra>",
    ))

    fig.update_layout(
        title=f"SVM — train={train_len}, test={test_len}, C={C}",
        barmode="group",
        xaxis=dict(title="Model", tickangle=-45),
        yaxis=dict(title="Accuracy", range=[0.0, 1.0]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1.0),
        margin=dict(l=40, r=20, t=60, b=80),
        template="plotly_white",
    )

    fig.write_html(save_path, include_plotlyjs="cdn")
    print(f"[plot] interactive HTML saved at {save_path}")


if __name__ == "__main__":
    num_qubits = 4
    C = 0.1
    train_len = 30
    test_len = 10
    num_workers = 4
    CKPT_DIR = "epoch1000_models/worker_pid3417964_20250906_045352"
    filename = "svm_results.html"

    X_train, X_test, y_train, y_test = data_load_and_process(reduction_sz=num_qubits,
                                                             train_len=train_len, test_len=test_len,
                                                             train_start=5000, test_start=1000)

    kernel_zz = make_overlap_kernel(zz_feature_map, num_qubits=num_qubits)
    za, zb = fit_eval_svm(X_train, y_train, X_test, y_test, kernel_zz, C=C)
    print(f"[BASE ZZ ] train={za:.6f}, test={zb:.6f}")

    ok, err = run_parallel_ckpts(
        ckpt_dir=CKPT_DIR,
        num_workers=num_workers,
        num_qubits=num_qubits,
        C=C,
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
    )

    baseline = ("(builtin)", "ZZ-FMAP", float(za), float(zb))
    ok_with_base = ok + [baseline]

    # 결과 출력
    plot_bar_accuracies(ok_results=ok_with_base, C=C, train_len=train_len, test_len=test_len, save_path=filename)
