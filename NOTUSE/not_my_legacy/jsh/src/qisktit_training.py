from src.model import GPT, GPTConfig
import torch
import numpy as np
from torch.nn import functional as F
from Qiskit.Circuit_generate import circuit_generator, get_tr_dist_matrix, get_fid_matrix, calculate_fidelity
from src.dataload import get_data
from torch.utils.data import DataLoader, TensorDataset
from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import Statevector, DensityMatrix, state_fidelity

X_train, y_train = get_data(name='iris')

n,p = X_train.shape

"""If use qubit matching, set num_qubits = p  otherwise, it is up to user's choice"""
num_qubits = 4
op_pool_size = p*7*num_qubits
max_gate = 10           # senario legth

class GPTQE(GPT):
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

    def calculate_loss(self, tokens, fidelity):
        """
        tokens: 모델이 생성한 시퀀스 토큰 텐서 (batch_size, sequence_length).
        fidelity: 우리가 정의한 목표 Fidelity 값.
        """
        # 현재 토큰 시퀀스와 다음 토큰 시퀀스를 분리
        current_tokens, next_tokens = tokens[:-1], tokens[1:]
    
        # 현재 토큰 시퀀스에 대한 예측 로짓 계산
        logits = self(current_tokens)
    
        # 실제 다음 토큰에 해당하는 로짓 선택
        next_token_mask = torch.nn.functional.one_hot(
            next_tokens, num_classes=self.config.vocab_size
        )
        
        next_token_logits = (logits * next_token_mask).sum(axis=2)
    
        # 각 시퀀스에 대한 누적 로짓 계산
        cumsum_logits = torch.cumsum(next_token_logits, dim=1)
    
        # 누적 로짓과 Fidelity 간의 차이를 제곱해 평균 손실로 계산
        loss = torch.mean(torch.square(cumsum_logits - fidelity))
    
        return loss


    @torch.no_grad()
    def generate(self, n_sequences, max_new_tokens, temperature=1.0, device="cpu"):
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
    
    @torch.no_grad()
    def evaluation(self, max_new_tokens, device="cpu"):
        idx = torch.zeros(size=(1), dtype=int, device=device)
        for _ in range(max_new_tokens):
            idx_cond = (idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]).to(device)
            logits = self(idx_cond).to(device)
            # pluck the logits at the final step
            logits = logits[:, -1, :]
            # set the logit of the first token so that its probability will be zero
            logits[:, 0] = float("inf")
            idx_next = torch.argmin(logits, dim=-1, keepdim=True)
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx


gpt = GPTQE(GPTConfig(
    vocab_size=op_pool_size + 1,
    block_size=max_gate,
    dropout=0.2,
    bias=False
)).to("cuda")

### Hyperparameter setting
opt = gpt.configure_optimizers(
    weight_decay=0.01, learning_rate=5e-5, betas=(0.9, 0.999), device_type="cuda"
)

num_iterations = 10000       # 총 학습 반복 횟수
loss_record = []

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

# TensorDataset을 사용하여 데이터셋 생성
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

# DataLoader로 배치 단위로 데이터 로드
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

for iteration in range(num_iterations):
    # 1. 모델이 회로를 생성할 토큰 시퀀스 생성
    tokens, logits = gpt.generate(n_sequences=1, max_new_tokens=max_gate, device="cuda") 
    tokens = tokens.reshape(-1,1)

    # 2. 생성된 토큰 시퀀스를 이용해 양자 회로를 생성하고 데이터를 임베딩
    quantum_states = []
    for x in X_train:
        circuit, state = circuit_generator(tokens, n_qubits=num_qubits, data=x) 
        quantum_states.append(state)
    
    # 3. Fidelity 계산
    fid_mat = get_fid_matrix(quantum_states)
    fidloss = calculate_fidelity(fid_mat, y_train)  # Fidelity 함수는 별도로 정의되어 있음

    # 5. 역전파 및 옵티마이저 업데이트
    opt.zero_grad()
    loss = gpt.calculate_loss(tokens, fidloss)
    loss.backward()
    opt.step()
    loss_record.append(loss.item())

    # 6. 진행 상황 출력
    if (iteration + 1) % 100 == 0:
        print(f"Iteration [{iteration + 1}/{num_iterations}], Loss: {loss.item()}")

print("Training completed.")