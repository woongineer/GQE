import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit.quantum_info import Statevector, DensityMatrix, state_fidelity
import torch


def gate_generate(circuit, num_qubits, qubit_idx, gate_idx, angle):
    """
    설명
    """
    angle = float(angle.item()) if isinstance(angle, torch.Tensor) else float(angle)  # 파라미터를 float으로 변환

    if gate_idx == 0:
        pass
    elif gate_idx == 1:
        circuit.rx(angle, qubit_idx)
    elif gate_idx == 2:
        circuit.ry(angle, qubit_idx)
    elif gate_idx == 3:
        circuit.rz(angle, qubit_idx)
    elif gate_idx == 4:
        circuit.rxx(angle, qubit_idx, (qubit_idx + 1) % num_qubits)
    elif gate_idx == 5:
        circuit.ryy(angle, qubit_idx, (qubit_idx + 1) % num_qubits)
    elif gate_idx == 6:
        circuit.rzz(angle, qubit_idx, (qubit_idx + 1) % num_qubits)


def circuit_generator(token_seq, n_qubits, data, matching = False):
    # Get token sequence and transform it (qubit, gate) index pair.
    backend = Aer.get_backend('statevector_simulator')
    pairs = []
    
    if matching:
        for token in token_seq:
            token = token - 1
            if token != -1:
                qubit_idx = token // 7
                gate_idx = token % 7
                pairs.append((qubit_idx, gate_idx))
    
    # Using this pair set, generate Quantum circuit
        circuit = QuantumCircuit(n_qubits)
        for q,g in pairs:
            q = int(q)  
            g = int(g)  
            gate_generate(circuit, n_qubits, q, g, angle=data[q])

    else:
        for token in token_seq:
            token = token - 1  
            if token != -1:
                data_idx = token // (n_qubits * 7)
                remainder = token % (n_qubits * 7)
                qubit_idx = remainder // 7
                gate_idx = remainder % 7
                pairs.append((data_idx, qubit_idx, gate_idx))
    
    # Using this pair set, generate Quantum circuit
        circuit = QuantumCircuit(n_qubits)
        for data_idx, q, g in pairs:
            data_idx, q, g = int(data_idx), int(q), int(g)
            gate_generate(circuit, n_qubits, q, g, angle=data[data_idx])
    
    state = Statevector.from_instruction(circuit)
    

    return circuit, state

def get_tr_dist_matrix(states):
    n = len(states)
    tr_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
        # state -> density matrix
            rho = DensityMatrix(states[i])
            sigma = DensityMatrix(states[j])
        
        # Get Trace Distance and store
            trace_dist = rho.distance(sigma, trace_distance_metric='trace')
            tr_mat[i, j] = trace_dist
            tr_mat[j, i] = trace_dist

    return tr_mat

def get_fid_matrix(states):
    n = len(states)
    fid_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
        # get fidelity
            fid = state_fidelity(states[i], states[j])
            fid_mat[i, j] = fid
            fid_mat[j, i] = fid

    return fid_mat

def calculate_fidelity(fidelity_matrix, labels):
    fidloss_list = []
    n = fidelity_matrix.shape[0]

    for i in range(n):
        for j in range(i+1, n):
            f = (fidelity_matrix[i,j]**2 - 0.5*(1+labels[i]*labels[j]))**2
            fidloss_list.append(f)

    return sum(fidloss_list) / len(fidloss_list)