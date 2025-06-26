import pennylane as qml
from pennylane import numpy as np


def QuantumEmbedding(tokens, data, n_qubits=2):
    tokens = tokens[1:]
    for token in tokens:
        token = token - 1
        feature = token // 15
        index = token % 15
        if 0 <= index < 3:
            # 방법 1: 인덱스 0~2 -> Pauli rotation gate 선택 후 CNOT 적용
            if index == 0:
                qml.RX(data[feature], wires=0)
                qml.RX(data[feature], wires=1)
            elif index == 1:
                qml.RY(data[feature], wires=0)
                qml.RY(data[feature], wires=1)
            elif index == 2:
                qml.RZ(data[feature], wires=0)
                qml.RZ(data[feature], wires=1)
            qml.CNOT(wires=[0, 1])

        elif 3 <= index < 6:
            # 방법 2: 인덱스 3~5 -> Hadamard, CZ, 그리고 Pauli rotation gate 적용
            qml.Hadamard(wires=0)
            qml.Hadamard(wires=1)
            qml.CZ(wires=[0, 1])
            if index == 3:
                qml.RX(data[feature], wires=0)
                qml.RX(data[feature], wires=1)
            elif index == 4:
                qml.RY(data[feature], wires=0)
                qml.RY(data[feature], wires=1)
            elif index == 5:
                qml.RZ(data[feature], wires=0)
                qml.RZ(data[feature], wires=1)

        elif 6 <= index < 15:
            # 방법 3: 인덱스 6~14 -> 두 큐빗에 Pauli rotation gate, 이후 제어 Pauli rotation gate 적용
            # pauli_index: 0->Rx, 1->Ry, 2->Rz; ctrl_index: 0->CRX, 1->CRY, 2->CRZ
            pauli_index = (index - 6) // 3
            ctrl_index = (index - 6) % 3
            
            if pauli_index == 0:
                qml.RX(data[feature], wires=0)
                qml.RX(data[feature], wires=1)
            elif pauli_index == 1:
                qml.RY(data[feature], wires=0)
                qml.RY(data[feature], wires=1)
            elif pauli_index == 2:
                qml.RZ(data[feature], wires=0)
                qml.RZ(data[feature], wires=1)
            
            if ctrl_index == 0:
                qml.CRX(data[feature], wires=[0, 1])
            elif ctrl_index == 1:
                qml.CRY(data[feature], wires=[0, 1])
            elif ctrl_index == 2:
                qml.CRZ(data[feature], wires=[0, 1])
    
