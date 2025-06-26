from src.model import GPT, GPTConfig
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from Pennylane.Circuit_generate import gate_generate, QuantumEmbedding
from src.dataload import get_data
from torch.utils.data import DataLoader, TensorDataset
import pennylane as qml

X_train, y_train = get_data(name='iris')

n,p = X_train.shape

"""If use qubit matching, set num_qubits = p  otherwise, it is up to user's choice"""
num_qubits = 4
op_pool_size = p*7*num_qubits
max_gate = 10           # senario legth

# TensorDataset을 사용하여 데이터셋 생성
train_dataset = TensorDataset(X_train, y_train)

# DataLoader로 배치 단위로 데이터 로드
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)


dev = qml.device('default.qubit', wires=num_qubits)

@qml.qnode(dev, interface='torch')
def quantum_circuit(tokens, data):
    QuantumEmbedding(tokens, num_qubits, data = data, matching=False)
    return qml.state()



class Transformer(GPT):
    def forward(self, idx):
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits
    
    def generate(self, n_sequences, max_new_tokens, temperature=1., device="cpu"):
        idx = torch.zeros(size=(n_sequences, 1), dtype=int, device=device)
        total_logits = torch.zeros(size=(n_sequences, 1), device=device)
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits = self(idx_cond)
            # pluck the logits at the final step
            logits = logits[:, -1, :]
            # set the logit of the first token so that its probability will be zero
            logits[:, 0] = float("inf")
            # apply softmax to convert logits to (normalized) probabilities and scale by desired temperature
            probs = F.softmax(-logits / temperature, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # # Accumulate logits
            total_logits += torch.gather(logits, index=idx_next, dim=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
        return idx, total_logits
    

class GNQE(nn.module):
    def __init__(self, vocab_size, block_size, dropout, n_layer, n_head, n_embd, bias):
        super(GNQE, self).__init__()
        self.transformer_decoder = Transformer((GPTConfig(
            vocab_size=vocab_size,
            block_size=block_size,
            dropout=dropout,
            n_layer = n_layer,
            n_head = n_head,
            n_embd = n_embd,
            bias=False)
            ))
        
        self.quantum_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes={})
    
    def gpt_forward(self, idx):
        logit = self.transformer_decoder.forward(idx)
        return logit

    def token_generate(self, max_new_tokens, device):
        tokens, logit = self.transformer_decoder.generate(n_sequences = 1, max_new_tokens = max_new_tokens, deviec=device)
        return tokens, logit
    
    def Qembedding(self, tokens, data):
        state = self.quantum_layer(torch.squeeze(tokens), data = data)
        return state
    


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GNQE(GPTConfig(
    vocab_size=op_pool_size + 1,
    block_size=max_gate,
    dropout=0.2,
    n_layer = 12,
    n_head = 12,
    n_embd=768,
    bias=False
)).to('cuda')
opt = torch.optim.AdamW(model.parameters())
n_epochs = 1000
loss_history = []

for epoch in range(n_epochs):
    # 학습 루프 내에서 손실 계산
    for X_batch, y_batch in train_loader:
        opt.zero_grad()
        batch_size = X_batch.size(0)

        # 배치의 상태 벡터 계산
        states = []
        token, logit = model.token_generate()
        for i in range(batch_size):
            state = model.Qembedding(X_batch[i])
            states.append(state)
        states = torch.stack(states)  # (batch_size, state_vector_size)

        # 상태 벡터 정규화
        states = states / torch.norm(states, dim=1, keepdim=True)

        # 상태 벡터의 켤레 복소수
        states_conj = torch.conj(states)

        # 내적 행렬 계산
        inner_products = torch.matmul(states_conj, states.T)  # (batch_size, batch_size)

        # 피델리티 행렬 계산
        fidelity_matrix = torch.abs(inner_products) ** 2

        # 라벨 곱 행렬 계산
        labels = y_batch.view(-1)  # (batch_size,)
        label_products = torch.outer(labels, labels)  # (batch_size, batch_size)

        # 손실 행렬 계산
        loss_matrix = (fidelity_matrix - 0.5 * (1 + label_products)) ** 2

        # 상삼각 행렬에서 i < j인 요소들 선택
        indices = torch.triu_indices(batch_size, batch_size, offset=1)
        loss_values = loss_matrix[indices[0], indices[1]]

        # 총 손실 계산
        loss = torch.mean((loss_values - logit)**2)
        
        loss_history.append(loss.item())
        # 역전파 및 옵티마이저 스텝
        loss.backward()
        opt.step()
        
    if (epoch+1) % 10 == 0 :
        print(f"Epoch : {epoch+1}, loss : {loss}")