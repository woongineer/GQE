import pennylane as qml
from pennylane import numpy as np
from sklearn.svm import SVC

from run_GQE.data import data_load_and_process

num_qubits = 4

dev = qml.device("default.qubit", wires=num_qubits)
WIRES = list(range(num_qubits))


def zz_feature_map(x, wires=WIRES):
    for i, w in enumerate(wires):
        qml.Hadamard(wires=w)
        qml.RZ(x[i], wires=w)

    n = len(wires)
    for i in range(n):
        for j in range(i + 1, n):
            qml.CNOT(wires=[wires[i], wires[j]])
            qml.RZ(2.0 * x[i] * x[j], wires=wires[j])
            qml.CNOT(wires=[wires[i], wires[j]])


def user_embedding(x, wires=WIRES):
    for i, w in enumerate(wires):
        qml.RY(x[i], wires=w)


def make_overlap_kernel(embedding_fn):
    @qml.qnode(dev)
    def overlap_circuit(x1, x2):
        embedding_fn(x1, wires=WIRES)
        qml.adjoint(embedding_fn)(x2, wires=WIRES)
        return qml.probs(wires=WIRES)

    def kernel_fn(x1, x2):
        return overlap_circuit(x1, x2)[0]

    return kernel_fn


kernel_user = make_overlap_kernel(user_embedding)
kernel_zz = make_overlap_kernel(zz_feature_map)


def fit_eval_svm(X_train, y_train, X_test, y_test, kernel_callable, C):
    K_train = qml.kernels.kernel_matrix(X_train, X_train, kernel_callable)
    clf = SVC(C=C, kernel="precomputed").fit(K_train, y_train)

    train_pred = clf.predict(K_train)
    train_acc = (train_pred == y_train).mean()

    K_test = qml.kernels.kernel_matrix(X_test, X_train, kernel_callable)
    test_pred = clf.predict(K_test)
    test_acc = (test_pred == y_test).mean()

    return train_acc, test_acc


if __name__ == "__main__":
    C = 0.1
    X_train, X_test, y_train, y_test = data_load_and_process("kmnist", reduction_sz=num_qubits, train_len=100, test_len=50)

    X_train = [np.array(x, requires_grad=False) for x in X_train]
    X_test = [np.array(x, requires_grad=False) for x in X_test]
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    zz_train_acc, zz_test_acc = fit_eval_svm(X_train, y_train, X_test, y_test, kernel_zz, C=C)
    print(f"[ZZ-EMB  ] SVM accuracy: train={zz_train_acc:.3f}, test={zz_test_acc:.3f}")

    print('dd')