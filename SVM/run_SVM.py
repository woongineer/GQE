import logging
import multiprocessing as mp
import os
import time
import traceback
from collections import defaultdict
from functools import partial
from logging.handlers import QueueHandler, QueueListener

import numpy as np
import numpy as pnp
import pennylane as qml
import plotly.graph_objects as go
import tensorflow as tf
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.svm import SVC

LOGGER_NAME = "svm_eval"
LOG_QUEUE = None


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


def _build_console_handler(level=logging.INFO):
    ch = logging.StreamHandler()
    ch.setLevel(level)
    fmt = logging.Formatter("[%(asctime)s][%(processName)s][%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S", )
    ch.setFormatter(fmt)
    return ch


def setup_main_logging(level=logging.INFO):
    log_queue: mp.Queue = mp.Queue(-1)
    console = _build_console_handler(level)

    listener = QueueListener(log_queue, console, respect_handler_level=True)
    listener.start()

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False
    logger.addHandler(QueueHandler(log_queue))
    return log_queue, listener


def setup_worker_logging(log_queue, level=logging.INFO):
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False
    logger.addHandler(QueueHandler(log_queue))
    return logger


def data_load_and_process(reduction_sz, train_len, test_len, train_start, test_start, data_path):
    logger = logging.getLogger(LOGGER_NAME)
    t0 = time.perf_counter()
    logger.info(f"[data] load KMNIST: path={data_path}, reduction={reduction_sz}, "
                f"train_len={train_len}, test_len={test_len}, "
                f"train_start={train_start}, test_start={test_start}")

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

    dt = time.perf_counter() - t0
    logger.info(f"[data] finished: train={len(x_train_out)}, test={len(x_test_out)}, took {dt:.2f}s")

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
    logger = logging.getLogger(LOGGER_NAME)
    t0 = time.perf_counter()
    logger.info(f"[svm] building kernel matrices: train={len(X_train)}, test={len(X_test)}, C={C}")
    K_train = qml.kernels.kernel_matrix(X_train, X_train, kernel_callable)
    clf = SVC(C=C, kernel="precomputed").fit(K_train, y_train)
    train_acc = clf.score(K_train, y_train)
    K_test = qml.kernels.kernel_matrix(X_test, X_train, kernel_callable)
    test_acc = clf.score(K_test, y_test)
    dt = time.perf_counter() - t0
    logger.info(f"[svm] done: train_acc={train_acc:.4f}, test_acc={test_acc:.4f}, took {dt:.2f}s")
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
    assert arch["type"] == "structured"
    gate_seq = arch["gate_seq"]
    num_bias = count_param_gates(gate_seq)
    net = build_single_input_bias_net(state, num_bias=num_bias, num_feat=num_feat)

    def user_embedding(x, wires):
        with torch.no_grad():
            xt = torch.tensor(x, dtype=torch.float32).view(1, -1)
            b = net(xt)[0].cpu().numpy()
        apply_structure(gate_seq, x, b)

    return user_embedding


def make_user_embedding_from_ckpt_no_bias(ckpt_path: str, num_feat: int):
    _, arch = load_bias_model(ckpt_path)
    assert arch["type"] == "structured"
    gate_seq = arch["gate_seq"]
    num_bias = count_param_gates(gate_seq)
    zero_bias = np.zeros((num_bias,), dtype=float)

    def user_embedding_no_bias(x, wires):
        apply_structure(gate_seq, x, zero_bias)

    return user_embedding_no_bias


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
    assert arch["type"] == "zz_nqe"
    n_layers = int(arch.get("n_layer", 1))
    enc = build_encoder_net(state, num_feat=num_feat)

    def zz_nqe_embedding(x, wires):
        with torch.no_grad():
            xt = torch.tensor(x, dtype=torch.float32).view(1, -1)
            z = enc(xt)[0].cpu().numpy()
        apply_zz_no_bias(n_layers, z, wires)

    return zz_nqe_embedding


def _worker_eval_ckpt(ckpt_file, ckpt_dir, num_qubits, C, X_train, X_test, y_train, y_test):
    _set_single_thread_env()

    logger = setup_worker_logging(LOG_QUEUE, level=logging.INFO)

    ckpt_path = os.path.join(ckpt_dir, ckpt_file)
    if os.path.basename(ckpt_path) == "zz.pt":
        return []

    outs = []
    try:
        logger.info(f"[worker] start ckpt: {ckpt_path}")
        t0 = time.perf_counter()
        _, arch = load_bias_model(ckpt_path)
        arch_type = arch.get("type", "structured")

        if arch_type == "structured":
            base_label = os.path.splitext(os.path.basename(ckpt_path))[0]  # G1, B3, ...

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

        else:
            logger.warning(f"[worker] unknown arch type '{arch_type}' in {ckpt_path}, skip.")
            return []

        dt = time.perf_counter() - t0
        logger.info(f"[worker] done ckpt: {ckpt_path}, results={outs}, took {dt:.2f}s")
        return outs

    except Exception as e:
        logger.error(f"[worker] error on {ckpt_path}: {e}\n{traceback.format_exc()}")
        return []


def eval_ckpt_dir(ckpt_dir, num_workers, num_qubits, C, X_train, X_test, y_train, y_test):
    files = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt") and f != "zz.pt"]
    files.sort()
    logger = logging.getLogger(LOGGER_NAME)
    logger.info(f"[dir] evaluate {len(files)} ckpts in: {ckpt_dir}")
    worker = partial(
        _worker_eval_ckpt,
        ckpt_dir=ckpt_dir,
        num_qubits=num_qubits, C=C,
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
    )
    results = []
    with mp.Pool(processes=num_workers, maxtasksperchild=1) as pool:
        for outs in pool.imap(worker, files, chunksize=1):
            if outs:
                results.extend(outs)
    logger.info(f"[dir] finished dir: {ckpt_dir}, collected {len(results)} result rows")
    return results


def run_pairwise(models_root, num_workers, num_qubits, C, data_path,
                 train_len, test_len, train_start0, test_start0, stride_train, stride_test):
    subdirs = [os.path.join(models_root, d) for d in os.listdir(models_root)
               if os.path.isdir(os.path.join(models_root, d))]
    subdirs.sort()
    n_pairs = len(subdirs)

    logger = logging.getLogger(LOGGER_NAME)
    logger.info(f"[run] models_root={models_root}, subdirs={n_pairs}, "
                f"num_qubits={num_qubits}, C={C}, workers={num_workers}")

    data_slices = make_data_slices_pairwise(
        n_pairs=n_pairs,
        train_len=train_len, test_len=test_len,
        train_start0=train_start0, test_start0=test_start0,
        stride_train=stride_train, stride_test=stride_test
    )

    agg_train = defaultdict(list)
    agg_test = defaultdict(list)

    for i, (ckpt_dir, sl) in enumerate(zip(subdirs, data_slices), start=1):
        logger.info(f"[run] ===== Pair {i}/{n_pairs} ===== dir={ckpt_dir} | slice={sl}")
        t_pair = time.perf_counter()

        # 데이터 로딩 (슬라이스 i)
        X_tr, X_te, y_tr, y_te = data_load_and_process(
            reduction_sz=num_qubits,
            train_len=sl["train_len"], test_len=sl["test_len"],
            train_start=sl["train_start"], test_start=sl["test_start"],
            data_path=data_path,
        )

        logger.info("[run] baseline ZZ-FMAP: start")
        t0 = time.perf_counter()
        kernel_zz = make_overlap_kernel(zz_feature_map, num_qubits=num_qubits)
        za, zb = fit_eval_svm(X_tr, y_tr, X_te, y_te, kernel_zz, C=C)
        agg_train["ZZ-FMAP"].append(float(za))
        agg_test["ZZ-FMAP"].append(float(zb))
        logger.info(f"[run] baseline ZZ-FMAP: done train={za:.4f}, test={zb:.4f}, took {time.perf_counter() - t0:.2f}s")

        res = eval_ckpt_dir(
            ckpt_dir=ckpt_dir,
            num_workers=num_workers,
            num_qubits=num_qubits,
            C=C,
            X_train=X_tr, X_test=X_te, y_train=y_tr, y_test=y_te
        )
        for label, tr, te in res:
            agg_train[label].append(tr)
            agg_test[label].append(te)

        logger.info(f"[run] ===== End Pair {i}/{n_pairs} ===== took {time.perf_counter() - t_pair:.2f}s")

    return agg_train, agg_test


def plot_errorbars(agg_train, agg_test, C, any_train_len, any_test_len, save_path="svm_errorbars.html"):
    logger = logging.getLogger(LOGGER_NAME)
    logger.info(f"[plot] start: {save_path}")
    t0 = time.perf_counter()
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
    logger.info(f"[plot] saved: {save_path}, took {time.perf_counter() - t0:.2f}s")


if __name__ == "__main__":
    try:
        mp.set_start_method("fork")
    except RuntimeError:
        pass

    LOG_LEVEL = logging.INFO
    log_queue, log_listener = setup_main_logging(level=LOG_LEVEL)

    LOG_QUEUE = log_queue

    logger = logging.getLogger(LOGGER_NAME)
    logger.info("===== START RUN =====")
    t_global = time.perf_counter()

    try:
        # MODELS_ROOT = "SVM/epoch1000_models"
        # DATA_PATH = "GQE/kmnist"
        # output_path = "SVM/svm_errorbars.html"
        MODELS_ROOT = "epoch1000_models"
        DATA_PATH = "/Users/jwheo/Desktop/Y/NQE/Neural-Quantum-Embedding/rl/kmnist"
        output_path = "svm_errorbars.html"

        num_qubits = 4
        C = 0.1
        num_workers = 64

        # train_len = 500
        # test_len = 100
        train_len = 40
        test_len = 10
        train_start0, test_start0 = 2000, 100
        stride_train, stride_test = 500, 100

        agg_train, agg_test = run_pairwise(
            models_root=MODELS_ROOT,
            num_workers=num_workers,
            num_qubits=num_qubits,
            C=C,
            data_path=DATA_PATH,
            train_len=train_len, test_len=test_len,
            train_start0=train_start0, test_start0=test_start0,
            stride_train=stride_train, stride_test=stride_test
        )

        plot_errorbars(agg_train, agg_test, C=C, any_train_len=train_len, any_test_len=test_len, save_path=output_path)

        logger.info(f"===== END RUN ===== took {time.perf_counter() - t_global:.2f}s")

    finally:
        # 리스너 정리
        try:
            log_listener.stop()
        except Exception:
            pass
