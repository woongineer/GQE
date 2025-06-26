import pennylane as qml
from pennylane import numpy as np
import torch

def RXX(x, wires, inverse=False):
  if inverse == False:
    qml.CNOT(wires=wires)
    qml.RX(x, wires=[wires[0]])
    qml.CNOT(wires=wires)


def RYY(x, wires, inverse=False):
  if inverse == False:
    qml.CY(wires=wires),
    qml.RY(x, wires=[wires[0]])
    qml.CY(wires=wires)


def RZZ(x, wires, inverse=False):
  if inverse == False:
    qml.CNOT(wires=wires)
    qml.RZ(x, wires=[wires[1]])
    qml.CNOT(wires=wires)
  

def gate_generate(num_qubits, qubit_idx, gate_idx, angle):
    """
    설명
    """
    if gate_idx == 0:
        qml.RX(angle, qubit_idx)
    elif gate_idx == 1:
        qml.RY(angle, qubit_idx)
    elif gate_idx == 2:
        qml.RZ(angle, qubit_idx)
    elif gate_idx == 3:
        qml.CNOT(wires=[qubit_idx, (qubit_idx+1) % num_qubits])

def QuantumEmbedding(token, n_qubits, data, matching=False):
  pairs = []
  if matching:
        for token in token:
            token = token - 1
            if token != -1:
                qubit_idx = token // 4
                gate_idx = token % 4
                pairs.append((qubit_idx, gate_idx))
    
    # Using this pair set, generate Quantum circuit
        for q,g in pairs:
            q = int(q)  
            g = int(g)  
            gate_generate(n_qubits, q, g, angle=data[q])

  else:
      for token in token:
          token = token - 1  
          if token != -1:
              data_idx = token // (n_qubits * 4)
              remainder = token % (n_qubits * 4)
              qubit_idx = remainder // 4
              gate_idx = remainder % 4
              pairs.append((data_idx, qubit_idx, gate_idx))
      for data_idx, q, g in pairs:
        data_idx, q, g = int(data_idx), int(q), int(g)
        gate_generate(n_qubits, q, g, angle=data[data_idx])