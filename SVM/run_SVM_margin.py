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
    ###########수정된 부분##########
    # probability=True 로 확률/확신도와 함께, decision_function 으로 margin도 같이 가져옴
    clf = SVC(C=C, kernel="precomputed", probability=True).fit(K_train, y_train)
    ###########수정된 부분##########
    train_acc = clf.score(K_train, y_train)
    K_test = qml.kernels.kernel_matrix(X_test, X_train, kernel_callable)
    test_acc = clf.score(K_test, y_test)

    ###########수정된 부분##########
    # ---- (1) CONFIDENCE: 맞춘 샘플의 max(prob) -> 2p-1 ----
    pred_tr = clf.predict(K_train)
    prob_tr = clf.predict_proba(K_train)
    y_tr_np = np.asarray(y_train).astype(int)
    correct_idx_tr = (pred_tr == y_tr_np)
    p_pred_tr = prob_tr.max(axis=1)
    conf_tr_correct = (2.0 * p_pred_tr - 1.0)[correct_idx_tr]

    pred_te = clf.predict(K_test)
    prob_te = clf.predict_proba(K_test)
    y_te_np = np.asarray(y_test).astype(int)
    correct_idx_te = (pred_te == y_te_np)
    p_pred_te = prob_te.max(axis=1)
    conf_te_correct = (2.0 * p_pred_te - 1.0)[correct_idx_te]

    corr_count_tr = int(correct_idx_tr.sum())
    corr_mean_tr  = float(np.mean(conf_tr_correct)) if conf_tr_correct.size else float("nan")
    corr_min_tr   = float(np.min(conf_tr_correct))  if conf_tr_correct.size else float("nan")
    corr_max_tr   = float(np.max(conf_tr_correct))  if conf_tr_correct.size else float("nan")

    corr_count_te = int(correct_idx_te.sum())
    corr_mean_te  = float(np.mean(conf_te_correct)) if conf_te_correct.size else float("nan")
    corr_min_te   = float(np.min(conf_te_correct))  if conf_te_correct.size else float("nan")
    corr_max_te   = float(np.max(conf_te_correct))  if conf_te_correct.size else float("nan")

    # ---- (2) MARGIN: decision_function 의 signed score (y∈{0,1} -> y_pm1=2y-1) ----
    f_tr = clf.decision_function(K_train)  # shape (N_tr,)
    f_te = clf.decision_function(K_test)   # shape (N_te,)
    y_tr_pm1 = (2 * y_tr_np - 1).astype(float)
    y_te_pm1 = (2 * y_te_np - 1).astype(float)
    m_tr = y_tr_pm1 * f_tr
    m_te = y_te_pm1 * f_te

    # "맞춘 샘플만"의 margin 통계
    m_tr_corr = m_tr[correct_idx_tr]
    m_te_corr = m_te[correct_idx_te]

    marg_count_tr = int(m_tr_corr.size)
    marg_mean_tr  = float(np.mean(m_tr_corr)) if m_tr_corr.size else float("nan")
    marg_min_tr   = float(np.min(m_tr_corr))  if m_tr_corr.size else float("nan")
    marg_max_tr   = float(np.max(m_tr_corr))  if m_tr_corr.size else float("nan")

    marg_count_te = int(m_te_corr.size)
    marg_mean_te  = float(np.mean(m_te_corr)) if m_te_corr.size else float("nan")
    marg_min_te   = float(np.min(m_te_corr))  if m_te_corr.size else float("nan")
    marg_max_te   = float(np.max(m_te_corr))  if m_te_corr.size else float("nan")

    # 반환: accuracy + confidence 통계 + margin 통계
    return (train_acc, test_acc,
            corr_count_tr, corr_mean_tr, corr_min_tr, corr_max_tr,
            corr_count_te, corr_mean_te, corr_min_te, corr_max_te,
            marg_count_tr, marg_mean_tr, marg_min_tr, marg_max_tr,
            marg_count_te, marg_mean_te, marg_min_te, marg_max_te)
    ###########수정된 부분##########


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
        ###########수정된 부분##########
        (tr_b, te_b,
         cnt_tr_b, mean_tr_b, min_tr_b, max_tr_b,
         cnt_te_b, mean_te_b, min_te_b, max_te_b,
         mcnt_tr_b, mmean_tr_b, mmin_tr_b, mmax_tr_b,
         mcnt_te_b, mmean_te_b, mmin_te_b, mmax_te_b) = \
            fit_eval_svm(X_train, y_train, X_test, y_test, kernel_bias, C=C)
        outs.append((f"{base_label}_Bias",
                     float(tr_b), float(te_b),
                     int(cnt_tr_b), float(mean_tr_b), float(min_tr_b), float(max_tr_b),
                     int(cnt_te_b), float(mean_te_b), float(min_te_b), float(max_te_b),
                     int(mcnt_tr_b), float(mmean_tr_b), float(mmin_tr_b), float(mmax_tr_b),
                     int(mcnt_te_b), float(mmean_te_b), float(mmin_te_b), float(mmax_te_b)))
        ###########수정된 부분##########

        user_emb_plain = make_user_embedding_from_ckpt_no_bias(ckpt_path, num_feat=num_qubits)
        kernel_plain = make_overlap_kernel(user_emb_plain, num_qubits=num_qubits)
        ###########수정된 부분##########
        (tr_p, te_p,
         cnt_tr_p, mean_tr_p, min_tr_p, max_tr_p,
         cnt_te_p, mean_te_p, min_te_p, max_te_p,
         mcnt_tr_p, mmean_tr_p, mmin_tr_p, mmax_tr_p,
         mcnt_te_p, mmean_te_p, mmin_te_p, mmax_te_p) = \
            fit_eval_svm(X_train, y_train, X_test, y_test, kernel_plain, C=C)
        outs.append((f"{base_label}",
                     float(tr_p), float(te_p),
                     int(cnt_tr_p), float(mean_tr_p), float(min_tr_p), float(max_tr_p),
                     int(cnt_te_p), float(mean_te_p), float(min_te_p), float(max_te_p),
                     int(mcnt_tr_p), float(mmean_tr_p), float(mmin_tr_p), float(mmax_tr_p),
                     int(mcnt_te_p), float(mmean_te_p), float(mmin_te_p), float(mmax_te_p)))
        ###########수정된 부분##########

    elif arch_type == "zz_nqe":
        zznqe_emb = make_zz_nqe_embedding_from_ckpt(ckpt_path, num_feat=num_qubits)
        kernel = make_overlap_kernel(zznqe_emb, num_qubits=num_qubits)
        ###########수정된 부분##########
        (tr, te,
         cnt_tr, mean_tr, min_tr, max_tr,
         cnt_te, mean_te, min_te, max_te,
         mcnt_tr, mmean_tr, mmin_tr, mmax_tr,
         mcnt_te, mmean_te, mmin_te, mmax_te) = \
            fit_eval_svm(X_train, y_train, X_test, y_test, kernel, C=C)
        outs.append(("zz_nqe",
                     float(tr), float(te),
                     int(cnt_tr), float(mean_tr), float(min_tr), float(max_tr),
                     int(cnt_te), float(mean_te), float(min_te), float(max_te),
                     int(mcnt_tr), float(mmean_tr), float(mmin_tr), float(mmax_tr),
                     int(mcnt_te), float(mmean_te), float(mmin_te), float(mmax_te)))
        ###########수정된 부분##########

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

    # confidence(correct-only)
    agg_corr_count_tr = defaultdict(list)
    agg_conf_mean_tr  = defaultdict(list)
    agg_conf_min_tr   = defaultdict(list)
    agg_conf_max_tr   = defaultdict(list)

    agg_corr_count_te = defaultdict(list)
    agg_conf_mean_te  = defaultdict(list)
    agg_conf_min_te   = defaultdict(list)
    agg_conf_max_te   = defaultdict(list)

    ###########수정된 부분##########
    # margin(correct-only)
    agg_marg_count_tr = defaultdict(list)
    agg_marg_mean_tr  = defaultdict(list)
    agg_marg_min_tr   = defaultdict(list)
    agg_marg_max_tr   = defaultdict(list)

    agg_marg_count_te = defaultdict(list)
    agg_marg_mean_te  = defaultdict(list)
    agg_marg_min_te   = defaultdict(list)
    agg_marg_max_te   = defaultdict(list)
    ###########수정된 부분##########

    for i, (ckpt_dir, sl) in enumerate(zip(subdirs, data_slices), start=1):
        logger.info(f"====== Pair {i}/{n_pairs} ====== dir={ckpt_dir} | slice={sl}")
        t_pair = time.perf_counter()

        X_tr, X_te, y_tr, y_te = data_load_and_process(reduction_sz=num_qubits,
                                                       train_len=sl["train_len"], test_len=sl["test_len"],
                                                       train_start=sl["train_start"], test_start=sl["test_start"],
                                                       data_path=data_path)

        t0 = time.perf_counter()
        kernel_zz = make_overlap_kernel(zz_apply_default, num_qubits=num_qubits)
        (zz_train, zz_test,
         zzcnt_tr, zzmean_tr, zzmin_tr, zzmax_tr,
         zzcnt_te, zzmean_te, zzmin_te, zzmax_te,
         zzm_cnt_tr, zzm_mean_tr, zzm_min_tr, zzm_max_tr,
         zzm_cnt_te, zzm_mean_te, zzm_min_te, zzm_max_te) = \
            fit_eval_svm(X_tr, y_tr, X_te, y_te, kernel_zz, C=C)
        agg_train["zz"].append(float(zz_train))
        agg_test["zz"].append(float(zz_test))

        # confidence
        agg_corr_count_tr["zz"].append(int(zzcnt_tr))
        agg_conf_mean_tr["zz"].append(float(zzmean_tr))
        agg_conf_min_tr["zz"].append(float(zzmin_tr))
        agg_conf_max_tr["zz"].append(float(zzmax_tr))
        agg_corr_count_te["zz"].append(int(zzcnt_te))
        agg_conf_mean_te["zz"].append(float(zzmean_te))
        agg_conf_min_te["zz"].append(float(zzmin_te))
        agg_conf_max_te["zz"].append(float(zzmax_te))

        # margin
        ###########수정된 부분##########
        agg_marg_count_tr["zz"].append(int(zzm_cnt_tr))
        agg_marg_mean_tr["zz"].append(float(zzm_mean_tr))
        agg_marg_min_tr["zz"].append(float(zzm_min_tr))
        agg_marg_max_tr["zz"].append(float(zzm_max_tr))
        agg_marg_count_te["zz"].append(int(zzm_cnt_te))
        agg_marg_mean_te["zz"].append(float(zzm_mean_te))
        agg_marg_min_te["zz"].append(float(zzm_min_te))
        agg_marg_max_te["zz"].append(float(zzm_max_te))
        ###########수정된 부분##########

        logger.info(f"baseline zz: train={zz_train:.4f}, test={zz_test:.4f}, took {time.perf_counter() - t0:.2f}s")

        res = eval_ckpt_dir(ckpt_dir=ckpt_dir, num_workers=num_workers, num_qubits=num_qubits, C=C, include=include,
                            X_train=X_tr, X_test=X_te, y_train=y_tr, y_test=y_te)
        for (label, tr, te,
             cnt_tr, mean_tr, min_tr, max_tr,
             cnt_te, mean_te, min_te, max_te,
             mcnt_tr, mmean_tr, mmin_tr, mmax_tr,
             mcnt_te, mmean_te, mmin_te, mmax_te) in res:
            agg_train[label].append(tr)
            agg_test[label].append(te)

            # confidence
            agg_corr_count_tr[label].append(int(cnt_tr))
            agg_conf_mean_tr[label].append(float(mean_tr))
            agg_conf_min_tr[label].append(float(min_tr))
            agg_conf_max_tr[label].append(float(max_tr))
            agg_corr_count_te[label].append(int(cnt_te))
            agg_conf_mean_te[label].append(float(mean_te))
            agg_conf_min_te[label].append(float(min_te))
            agg_conf_max_te[label].append(float(max_te))

            # margin
            ###########수정된 부분##########
            agg_marg_count_tr[label].append(int(mcnt_tr))
            agg_marg_mean_tr[label].append(float(mmean_tr))
            agg_marg_min_tr[label].append(float(mmin_tr))
            agg_marg_max_tr[label].append(float(mmax_tr))
            agg_marg_count_te[label].append(int(mcnt_te))
            agg_marg_mean_te[label].append(float(mmean_te))
            agg_marg_min_te[label].append(float(mmin_te))
            agg_marg_max_te[label].append(float(mmax_te))
            ###########수정된 부분##########

        logger.info(f"====== End Pair {i}/{n_pairs} ====== took {time.perf_counter() - t_pair:.2f}s")

    ###########수정된 부분##########
    return (agg_train, agg_test,
            agg_corr_count_tr, agg_conf_mean_tr, agg_conf_min_tr, agg_conf_max_tr,
            agg_corr_count_te, agg_conf_mean_te, agg_conf_min_te, agg_conf_max_te,
            agg_marg_count_tr, agg_marg_mean_tr, agg_marg_min_tr, agg_marg_max_tr,
            agg_marg_count_te, agg_marg_mean_te, agg_marg_min_te, agg_marg_max_te)
    ###########수정된 부분##########


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
    logging.getLogger(logger_name).info(f"[plot] saved: {save_path}")


###########수정된 부분##########
def plot_correct_confidence(
    agg_corr_count_tr, agg_conf_mean_tr, agg_conf_min_tr, agg_conf_max_tr,
    agg_corr_count_te, agg_conf_mean_te, agg_conf_min_te, agg_conf_max_te,
    save_path="svm_correct_confidence.html"
):
    import numpy as _np
    labels = sorted(set(
        list(agg_corr_count_tr.keys()) + list(agg_corr_count_te.keys())
    ))

    cnt_tr = [int(_np.sum(agg_corr_count_tr[l])) if agg_corr_count_tr[l] else 0 for l in labels]
    cnt_te = [int(_np.sum(agg_corr_count_te[l])) if agg_corr_count_te[l] else 0 for l in labels]

    def _nanmean(x): return float(_np.nanmean(x)) if len(x) else float("nan")
    def _nanmin(x):  return float(_np.nanmin(x))  if len(x) else float("nan")
    def _nanmax(x):  return float(_np.nanmax(x))  if len(x) else float("nan")

    mean_tr = [_nanmean(agg_conf_mean_tr[l]) for l in labels]
    mean_te = [_nanmean(agg_conf_mean_te[l]) for l in labels]
    gmin_tr = [_nanmin(agg_conf_min_tr[l]) for l in labels]
    gmax_tr = [_nanmax(agg_conf_max_tr[l]) for l in labels]
    gmin_te = [_nanmin(agg_conf_min_te[l]) for l in labels]
    gmax_te = [_nanmax(agg_conf_max_te[l]) for l in labels]

    lower_tr = _np.maximum(_np.nan_to_num(_np.array(mean_tr)-_np.array(gmin_tr), nan=0.0), 0.0)
    upper_tr = _np.maximum(_np.nan_to_num(_np.array(gmax_tr)-_np.array(mean_tr), nan=0.0), 0.0)
    lower_te = _np.maximum(_np.nan_to_num(_np.array(mean_te)-_np.array(gmin_te), nan=0.0), 0.0)
    upper_te = _np.maximum(_np.nan_to_num(_np.array(gmax_te)-_np.array(mean_te), nan=0.0), 0.0)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=mean_tr, name="Train mean(2P-1)",
                         error_y=dict(type="data", array=upper_tr, arrayminus=lower_tr, visible=True, thickness=1.2),
                         customdata=[f"{mn:.4f} / {mx:.4f}" for mn, mx in zip(gmin_tr, gmax_tr)],
                         hovertemplate="Model: %{x}<br>Train mean: %{y:.4f}<br>Train min/max: %{customdata}<extra></extra>",
                         offsetgroup="conf_tr", marker_line_width=0, yaxis="y"))
    fig.add_trace(go.Bar(x=labels, y=mean_te, name="Test mean(2P-1)",
                         error_y=dict(type="data", array=upper_te, arrayminus=lower_te, visible=True, thickness=1.2),
                         customdata=[f"{mn:.4f} / {mx:.4f}" for mn, mx in zip(gmin_te, gmax_te)],
                         hovertemplate="Model: %{x}<br>Test mean: %{y:.4f}<br>Test min/max: %{customdata}<extra></extra>",
                         offsetgroup="conf_te", marker_line_width=0, yaxis="y"))
    fig.add_trace(go.Bar(x=labels, y=cnt_tr, name="Train correct count", yaxis="y2", opacity=0.45,
                         offsetgroup="cnt_tr", hovertemplate="Model: %{x}<br>Train correct: %{y}<extra></extra>",
                         marker_line_width=0))
    fig.add_trace(go.Bar(x=labels, y=cnt_te, name="Test correct count", yaxis="y2", opacity=0.45,
                         offsetgroup="cnt_te", hovertemplate="Model: %{x}<br>Test correct: %{y}<extra></extra>",
                         marker_line_width=0))

    fig.update_layout(
        title="Correct-only confidence (mean with min/max) and correct counts",
        xaxis=dict(title="Model", categoryorder="array", categoryarray=labels, tickangle=-20),
        yaxis=dict(title="Confidence (2P(y_true)-1)", range=[0.0, 1.0]),
        yaxis2=dict(title="Correct count", overlaying="y", side="right", rangemode="tozero"),
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        template="plotly_white",
        margin=dict(l=60, r=60, t=60, b=80),
    )

    fig.write_html(save_path, include_plotlyjs="cdn")
    logging.getLogger(logger_name).info(f"[plot] saved: {save_path}")


###########수정된 부분##########
def plot_correct_margin(
    agg_marg_count_tr, agg_marg_mean_tr, agg_marg_min_tr, agg_marg_max_tr,
    agg_marg_count_te, agg_marg_mean_te, agg_marg_min_te, agg_marg_max_te,
    save_path="svm_correct_margin.html"
):
    import numpy as _np
    labels = sorted(set(
        list(agg_marg_count_tr.keys()) + list(agg_marg_count_te.keys())
    ))

    # 카운트(정답 개수) 합계
    cnt_tr = [int(_np.sum(agg_marg_count_tr[l])) if agg_marg_count_tr[l] else 0 for l in labels]
    cnt_te = [int(_np.sum(agg_marg_count_te[l])) if agg_marg_count_te[l] else 0 for l in labels]

    def _nm(x, fn):
        return float(fn(x)) if len(x) else float("nan")

    mean_tr = [_nm(agg_marg_mean_tr[l], _np.nanmean) for l in labels]
    mean_te = [_nm(agg_marg_mean_te[l], _np.nanmean) for l in labels]
    gmin_tr = [_nm(agg_marg_min_tr[l],  _np.nanmin)  for l in labels]
    gmax_tr = [_nm(agg_marg_max_tr[l],  _np.nanmax)  for l in labels]
    gmin_te = [_nm(agg_marg_min_te[l],  _np.nanmin)  for l in labels]
    gmax_te = [_nm(agg_marg_max_te[l],  _np.nanmax)  for l in labels]

    # 에러바 길이(음수 방지)
    lower_tr = _np.maximum(_np.nan_to_num(_np.array(mean_tr)-_np.array(gmin_tr), nan=0.0), 0.0)
    upper_tr = _np.maximum(_np.nan_to_num(_np.array(gmax_tr)-_np.array(mean_tr), nan=0.0), 0.0)
    lower_te = _np.maximum(_np.nan_to_num(_np.array(mean_te)-_np.array(gmin_te), nan=0.0), 0.0)
    upper_te = _np.maximum(_np.nan_to_num(_np.array(gmax_te)-_np.array(mean_te), nan=0.0), 0.0)

    # y축 범위 자동 설정 (margin은 상한이 없음)
    all_vals = _np.array(
        [v for v in mean_tr+mean_te+gmin_tr+gmax_tr+gmin_te+gmax_te if not _np.isnan(v)],
        dtype=float
    )
    y_min = float(all_vals.min()) if all_vals.size else 0.0
    y_max = float(all_vals.max()) if all_vals.size else 1.0
    # 살짝 여유
    pad = 0.05 * (y_max - y_min if y_max > y_min else 1.0)
    y_min_plot = y_min - pad
    y_max_plot = y_max + pad

    fig = go.Figure()
    # Train margin (bar + error bars)
    fig.add_trace(go.Bar(
        x=labels, y=mean_tr, name="Train mean(margin)",
        error_y=dict(type="data", array=upper_tr, arrayminus=lower_tr, visible=True, thickness=1.2),
        customdata=[f"{mn:.4f} / {mx:.4f}" for mn, mx in zip(gmin_tr, gmax_tr)],
        hovertemplate="Model: %{x}<br>Train mean: %{y:.4f}<br>Train min/max: %{customdata}<extra></extra>",
        offsetgroup="marg_tr", marker_line_width=0, yaxis="y"
    ))
    # Test margin (bar + error bars)
    fig.add_trace(go.Bar(
        x=labels, y=mean_te, name="Test mean(margin)",
        error_y=dict(type="data", array=upper_te, arrayminus=lower_te, visible=True, thickness=1.2),
        customdata=[f"{mn:.4f} / {mx:.4f}" for mn, mx in zip(gmin_te, gmax_te)],
        hovertemplate="Model: %{x}<br>Test mean: %{y:.4f}<br>Test min/max: %{customdata}<extra></extra>",
        offsetgroup="marg_te", marker_line_width=0, yaxis="y"
    ))
    # Correct count (dual y-axis)
    fig.add_trace(go.Bar(
        x=labels, y=cnt_tr, name="Train correct count",
        yaxis="y2", opacity=0.45, offsetgroup="cnt_tr",
        hovertemplate="Model: %{x}<br>Train correct: %{y}<extra></extra>",
        marker_line_width=0
    ))
    fig.add_trace(go.Bar(
        x=labels, y=cnt_te, name="Test correct count",
        yaxis="y2", opacity=0.45, offsetgroup="cnt_te",
        hovertemplate="Model: %{x}<br>Test correct: %{y}<extra></extra>",
        marker_line_width=0
    ))

    fig.update_layout(
        title="Correct-only margin (mean with min/max) and correct counts",
        xaxis=dict(title="Model", categoryorder="array", categoryarray=labels, tickangle=-20),
        yaxis=dict(title="Margin (y_pm1 * f(x))"),
        yaxis2=dict(title="Correct count", overlaying="y", side="right", rangemode="tozero"),
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        template="plotly_white",
        margin=dict(l=60, r=60, t=60, b=80),
    )

    fig.write_html(save_path, include_plotlyjs="cdn")
    logging.getLogger(logger_name).info(f"[plot] saved: {save_path}")
###########수정된 부분##########

###########수정된 부분##########


if __name__ == "__main__":
    mp.set_start_method("fork")

    setup_logging(level=logging.INFO)
    logger = logging.getLogger(logger_name)
    logger.info("======== START RUN ========")
    t_global = time.perf_counter()

    model_root = "epoch1000_models"
    data_path = "/Users/jwheo/Desktop/Y/NQE/Neural-Quantum-Embedding/rl/kmnist"
    output_path = "svm_errorbars.html"
    confidence_output_path = "svm_correct_confidence.html"
    margin_output_path = "svm_correct_margin.html"


    num_qubits = 4
    C = 0.1
    num_workers = 6
    include = {"upto": 2, "model_type": ["zz_nqe", "G"]}

    train_len = 20
    test_len = 5
    train_start0, test_start0 = 2000, 100
    stride_train, stride_test = 500, 100

    (agg_train, agg_test,
     agg_cnt_tr, agg_cmean_tr, agg_cmin_tr, agg_cmax_tr,
     agg_cnt_te, agg_cmean_te, agg_cmin_te, agg_cmax_te,
     ###########수정된 부분##########
     agg_mcnt_tr, agg_mmean_tr, agg_mmin_tr, agg_mmax_tr,
     agg_mcnt_te, agg_mmean_te, agg_mmin_te, agg_mmax_te
     ###########수정된 부분##########
     ) = run_pairwise(
        models_root=model_root, num_workers=num_workers, num_qubits=num_qubits, C=C,
        data_path=data_path, include=include,
        train_len=train_len, test_len=test_len,
        train_start0=train_start0, test_start0=test_start0,
        stride_train=stride_train, stride_test=stride_test
    )

    plot_errorbars(agg_train, agg_test, C=C, any_train_len=train_len, any_test_len=test_len, save_path=output_path)

    # 맞춘 샘플 확신도 요약 (Train/Test 구분)
    for label in sorted(set(list(agg_cnt_tr.keys()) + list(agg_cnt_te.keys()))):
        cm_tr  = float(np.nanmean(agg_cmean_tr[label])) if agg_cmean_tr[label] else float("nan")
        cmin_tr = float(np.nanmean(agg_cmin_tr[label])) if agg_cmin_tr[label] else float("nan")
        cmax_tr = float(np.nanmean(agg_cmax_tr[label])) if agg_cmax_tr[label] else float("nan")
        cnt_tr  = int(np.sum(agg_cnt_tr[label])) if agg_cnt_tr[label] else 0
        logger.info(f"[correct-confidence][TRAIN] {label}: count_total={cnt_tr}, mean={cm_tr:.6f}, min={cmin_tr:.6f}, max={cmax_tr:.6f}")

        cm_te  = float(np.nanmean(agg_cmean_te[label])) if agg_cmean_te[label] else float("nan")
        cmin_te = float(np.nanmean(agg_cmin_te[label])) if agg_cmin_te[label] else float("nan")
        cmax_te = float(np.nanmean(agg_cmax_te[label])) if agg_cmax_te[label] else float("nan")
        cnt_te  = int(np.sum(agg_cnt_te[label])) if agg_cnt_te[label] else 0
        logger.info(f"[correct-confidence][TEST ] {label}: count_total={cnt_te}, mean={cm_te:.6f}, min={cmin_te:.6f}, max={cmax_te:.6f}")

    ###########수정된 부분##########
    # 맞춘 샘플 마진 요약 (Train/Test 구분)
    for label in sorted(set(list(agg_mcnt_tr.keys()) + list(agg_mcnt_te.keys()))):
        mm_tr  = float(np.nanmean(agg_mmean_tr[label])) if agg_mmean_tr[label] else float("nan")
        mmin_tr = float(np.nanmean(agg_mmin_tr[label])) if agg_mmin_tr[label] else float("nan")
        mmax_tr = float(np.nanmean(agg_mmax_tr[label])) if agg_mmax_tr[label] else float("nan")
        mcnt_tr = int(np.sum(agg_mcnt_tr[label])) if agg_mcnt_tr[label] else 0
        logger.info(f"[correct-margin][TRAIN] {label}: count_total={mcnt_tr}, mean={mm_tr:.6f}, min={mmin_tr:.6f}, max={mmax_tr:.6f}")

        mm_te  = float(np.nanmean(agg_mmean_te[label])) if agg_mmean_te[label] else float("nan")
        mmin_te = float(np.nanmean(agg_mmin_te[label])) if agg_mmin_te[label] else float("nan")
        mmax_te = float(np.nanmean(agg_mmax_te[label])) if agg_mmax_te[label] else float("nan")
        mcnt_te = int(np.sum(agg_mcnt_te[label])) if agg_mcnt_te[label] else 0
        logger.info(f"[correct-margin][TEST ] {label}: count_total={mcnt_te}, mean={mm_te:.6f}, min={mmin_te:.6f}, max={mmax_te:.6f}")
    ###########수정된 부분##########

    # confidence 시각화
    plot_correct_confidence(
        agg_cnt_tr, agg_cmean_tr, agg_cmin_tr, agg_cmax_tr,
        agg_cnt_te, agg_cmean_te, agg_cmin_te, agg_cmax_te,
        save_path=confidence_output_path
    )

    plot_correct_margin(
        agg_mcnt_tr, agg_mmean_tr, agg_mmin_tr, agg_mmax_tr,
        agg_mcnt_te, agg_mmean_te, agg_mmin_te, agg_mmax_te,
        save_path=margin_output_path
    )

    logger.info(f"======== END RUN ======== took {time.perf_counter() - t_global:.2f}s")
