import logging
import multiprocessing as mp
import os
import time
from collections import defaultdict
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


###################################### logging ######################################

logger_name = "svm_eval"


def set_single_thread_env():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
    torch.set_num_threads(1)


def setup_logging(level=logging.INFO):
    logging.basicConfig(level=level, format="[%(asctime)s][%(processName)s][%(levelname)s] %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S", force=True, )
    return logging.getLogger(logger_name)


###################################### data ######################################


def data_load_and_process(reduction_sz, train_len, test_len, train_start, test_start, data_path):
    logger = logging.getLogger(logger_name)
    logger.info(f"KMNIST: train/test from = {train_start}/{test_start}, train/test len = {train_len}/{test_len}")

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


def make_data_slices_pairwise(n_pairs, train_len, test_len, train_start0, test_start0, stride_train, stride_test):
    specs = []
    for k in range(n_pairs):
        specs.append({
            "train_start": train_start0 + k * stride_train,
            "test_start": test_start0 + k * stride_test,
            "train_len": train_len,
            "test_len": test_len,
        })
    return specs


###################################### SVM related ######################################


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


###################################### build/restore model ######################################


def load_ckpt(ckpt_path):
    model = torch.load(ckpt_path, map_location="cpu")
    return model["state_dict"], model["arch_info"]



def zz_apply_default(x, wires):
    n = len(wires)
    for i in range(n):
        qml.Hadamard(wires=wires[i])
        qml.RZ(x[i], wires=wires[i])
    for i in range(n):
        for j in range(i + 1, n):
            qml.CNOT(wires=[wires[i], wires[j]])
            qml.RZ(2.0 * x[i] * x[j], wires=wires[j])
            qml.CNOT(wires=[wires[i], wires[j]])


class BiasNet(nn.Module):
    def __init__(self, num_bias, num_feat):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Linear(num_feat, num_feat * 4), nn.ReLU(),
            nn.Linear(num_feat * 4, num_feat * 2), nn.ReLU(),
        )
        self.head = nn.Linear(num_feat * 2, num_bias, bias=True)

    def forward(self, x):
        z = self.feat(x)
        return self.head(z)


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
    state, arch = load_ckpt(ckpt_path)
    assert arch["type"] == "structured"
    gate_seq = arch["gate_seq"]
    num_bias = count_param_gates(gate_seq)
    net = build_model(BiasNet, state, num_bias=num_bias, num_feat=num_feat)

    def user_embedding(x, wires):
        with torch.no_grad():
            xt = torch.tensor(x, dtype=torch.float32).view(1, -1)
            b = net(xt)[0].cpu().numpy()
        apply_structure(gate_seq, x, b)

    return user_embedding


def make_user_embedding_from_ckpt_no_bias(ckpt_path: str, num_feat: int):
    _, arch = load_ckpt(ckpt_path)
    assert arch["type"] == "structured"
    gate_seq = arch["gate_seq"]
    num_bias = count_param_gates(gate_seq)
    zero_bias = np.zeros((num_bias,), dtype=float)

    def user_embedding_no_bias(x, wires):
        apply_structure(gate_seq, x, zero_bias)

    return user_embedding_no_bias


class NQE(nn.Module):
    def __init__(self, num_feat):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(num_feat, num_feat * 2), nn.ReLU(),
            nn.Linear(num_feat * 2, num_feat * 2), nn.ReLU(),
            nn.Linear(num_feat * 2, num_feat),
        )

    def forward(self, x):
        return self.enc(x)


def build_model(net_cls, state_dict: dict, **kwargs):
    net = net_cls(**kwargs)
    with torch.no_grad():
        for k, v in net.state_dict().items():
            v.copy_(state_dict[k])
    net.eval()
    return net



def zz_apply_nqe(n_layers: int, z, wires):
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
    state, arch = load_ckpt(ckpt_path)
    assert arch["type"] == "zz_nqe"
    n_layers = int(arch.get("n_layer", 1))
    net = build_model(NQE, state, num_feat=num_feat)

    def zz_nqe_embedding(x, wires):
        with torch.no_grad():
            xt = torch.tensor(x, dtype=torch.float32).view(1, -1)
            z = net(xt)[0].cpu().numpy()
        zz_apply_nqe(n_layers, z, wires)

    return zz_nqe_embedding


###################################### Runner ######################################


def worker_eval_ckpt(ckpt_file, ckpt_dir, num_qubits, C, X_train, X_test, y_train, y_test):
    set_single_thread_env()
    setup_logging(level=logging.INFO)
    logger = logging.getLogger(logger_name)
    t0 = time.perf_counter()

    ckpt_path = os.path.join(ckpt_dir, ckpt_file)
    outs = []

    _, arch = load_ckpt(ckpt_path)
    arch_type = arch.get("type", "structured")

    if arch_type == "structured":
        base_label = os.path.splitext(os.path.basename(ckpt_path))[0]

        user_emb_bias = make_user_embedding_from_ckpt(ckpt_path, num_feat=num_qubits)
        kernel_bias = make_overlap_kernel(user_emb_bias, num_qubits=num_qubits)
        tr_b, te_b = fit_eval_svm(X_train, y_train, X_test, y_test, kernel_bias, C=C)
        outs.append((f"{base_label}_Bias", float(tr_b), float(te_b)))

        user_emb_plain = make_user_embedding_from_ckpt_no_bias(ckpt_path, num_feat=num_qubits)
        kernel_plain = make_overlap_kernel(user_emb_plain, num_qubits=num_qubits)
        tr_p, te_p = fit_eval_svm(X_train, y_train, X_test, y_test, kernel_plain, C=C)
        outs.append((f"{base_label}", float(tr_p), float(te_p)))

    elif arch_type == "zz_nqe":
        zznqe_emb = make_zz_nqe_embedding_from_ckpt(ckpt_path, num_feat=num_qubits)
        kernel = make_overlap_kernel(zznqe_emb, num_qubits=num_qubits)
        tr, te = fit_eval_svm(X_train, y_train, X_test, y_test, kernel, C=C)
        outs.append(("zz_nqe", float(tr), float(te)))

    logger.info(f"ckpt: {ckpt_path}, results={outs}, took {time.perf_counter() - t0:.2f}s")
    return outs


def eval_ckpt_dir(ckpt_dir, num_workers, num_qubits, C, include, X_train, X_test, y_train, y_test):
    files_all = {f for f in os.listdir(ckpt_dir) if f.endswith(".pt")}
    model_types = include['model_type']
    upto = include['upto']
    wanted = []
    for mt in model_types:
        if mt == "zz_nqe":
            wanted.append("zz_nqe.pt")
        else:
            wanted.extend(f"{mt}{i}.pt" for i in range(1, upto + 1))
    files = sorted([f for f in wanted if f in files_all])

    worker = partial(worker_eval_ckpt, ckpt_dir=ckpt_dir, num_qubits=num_qubits, C=C,
                     X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    results = []
    with mp.Pool(processes=num_workers, maxtasksperchild=1) as pool:
        for outs in pool.imap(worker, files, chunksize=1):
            results += outs
    return results


def run_pairwise(models_root, num_workers, num_qubits, C, data_path, include,
                 train_len, test_len, train_start0, test_start0, stride_train, stride_test):
    subdirs = sorted(os.path.join(models_root, d) for d in os.listdir(models_root))
    n_pairs = len(subdirs)

    logger = logging.getLogger(logger_name)
    logger.info(f"models_root={models_root}, subdirs={n_pairs}, C={C}, workers={num_workers}")

    data_slices = make_data_slices_pairwise(
        n_pairs=n_pairs,
        train_len=train_len, test_len=test_len,
        train_start0=train_start0, test_start0=test_start0,
        stride_train=stride_train, stride_test=stride_test
    )

    agg_train = defaultdict(list)
    agg_test = defaultdict(list)

    for i, (ckpt_dir, sl) in enumerate(zip(subdirs, data_slices), start=1):
        logger.info(f"====== Pair {i}/{n_pairs} ====== dir={ckpt_dir} | slice={sl}")
        t_pair = time.perf_counter()

        X_tr, X_te, y_tr, y_te = data_load_and_process(reduction_sz=num_qubits,
                                                       train_len=sl["train_len"], test_len=sl["test_len"],
                                                       train_start=sl["train_start"], test_start=sl["test_start"],
                                                       data_path=data_path)

        t0 = time.perf_counter()
        kernel_zz = make_overlap_kernel(zz_apply_default, num_qubits=num_qubits)
        zz_train, zz_test = fit_eval_svm(X_tr, y_tr, X_te, y_te, kernel_zz, C=C)
        agg_train["zz"].append(float(zz_train))
        agg_test["zz"].append(float(zz_test))
        logger.info(f"baseline zz: train={zz_train:.4f}, test={zz_test:.4f}, took {time.perf_counter() - t0:.2f}s")

        res = eval_ckpt_dir(ckpt_dir=ckpt_dir, num_workers=num_workers, num_qubits=num_qubits, C=C, include=include,
                            X_train=X_tr, X_test=X_te, y_train=y_tr, y_test=y_te)
        for label, tr, te in res:
            agg_train[label].append(tr)
            agg_test[label].append(te)

        logger.info(f"====== End Pair {i}/{n_pairs} ====== took {time.perf_counter() - t_pair:.2f}s")

    return agg_train, agg_test


###################################### plot & utils ######################################


def count_param_gates(gate_seq):
    return sum(g[1] is not None for g in gate_seq)


def plot_errorbars(agg_train, agg_test, C, any_train_len, any_test_len, save_path="svm_errorbars.html"):
    logger = logging.getLogger(logger_name)
    labels = sorted(
        agg_test.keys(),
        key=lambda k: (np.mean(agg_test[k]) if len(agg_test[k]) > 0 else 0.0),
        reverse=True
    )

    train_means = [float(np.mean(agg_train[label])) for label in labels]
    train_stds = [float(np.std(agg_train[label], ddof=0)) for label in labels]
    test_means = [float(np.mean(agg_test[label])) for label in labels]
    test_stds = [float(np.std(agg_test[label], ddof=0)) for label in labels]

    xs = np.arange(len(labels), dtype=float)
    offset = 0.18

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=xs - offset,
        y=train_means,
        mode="markers",
        name="Train (mean ± std)",
        marker=dict(color="blue", size=9),
        error_y=dict(type="data", array=train_stds, visible=True, thickness=1.2),
        hovertemplate="Model: %{customdata}<br>Train mean: %{y:.6f}<br>Train std: %{meta:.6f}<extra></extra>",
        customdata=labels,
        meta=np.array(train_stds),
    ))

    fig.add_trace(go.Scatter(
        x=xs + offset,
        y=test_means,
        mode="markers",
        name="Test (mean ± std)",
        marker=dict(color="red", size=9),
        error_y=dict(type="data", array=test_stds, visible=True, thickness=1.2),
        hovertemplate="Model: %{customdata}<br>Test mean: %{y:.6f}<br>Test std: %{meta:.6f}<extra></extra>",
        customdata=labels,
        meta=np.array(test_stds),
    ))

    fig.update_layout(
        title=f"SVM (mean ± std over runs) — sample_train={any_train_len}, sample_test={any_test_len}, C={C}",
        xaxis=dict(
            title="Model",
            tickmode="array",
            tickvals=xs,
            ticktext=labels,
            tickangle=-45
        ),
        yaxis=dict(title="Accuracy", range=[0.0, 1.0]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        margin=dict(l=40, r=20, t=60, b=90),
        template="plotly_white",
    )

    fig.write_html(save_path, include_plotlyjs="cdn")
    logger.info(f"[plot] saved: {save_path}")


if __name__ == "__main__":
    mp.set_start_method("fork")

    setup_logging(level=logging.INFO)
    logger = logging.getLogger(logger_name)
    logger.info("======== START RUN ========")
    t_global = time.perf_counter()

    # MODELS_ROOT = "SVM/epoch1000_models"
    # DATA_PATH = "GQE/kmnist"
    # output_path = "SVM/svm_errorbars.html"
    model_root = "epoch1000_models"
    data_path = "/Users/jwheo/Desktop/Y/NQE/Neural-Quantum-Embedding/rl/kmnist"
    output_path = "svm_errorbars.html"

    num_qubits = 4
    C = 0.1
    # num_workers = 64
    num_workers = 6
    include = {"upto": 2, "model_type": ["zz_nqe", "G"]}

    # train_len = 500
    # test_len = 100
    train_len = 20
    test_len = 5
    train_start0, test_start0 = 2000, 100
    stride_train, stride_test = 500, 100

    agg_train, agg_test = run_pairwise(models_root=model_root, num_workers=num_workers, num_qubits=num_qubits, C=C,
                                       data_path=data_path, include=include,
                                       train_len=train_len, test_len=test_len,
                                       train_start0=train_start0, test_start0=test_start0,
                                       stride_train=stride_train, stride_test=stride_test)

    plot_errorbars(agg_train, agg_test, C=C, any_train_len=train_len, any_test_len=test_len, save_path=output_path)

    logger.info(f"======== END RUN ======== took {time.perf_counter() - t_global:.2f}s")
