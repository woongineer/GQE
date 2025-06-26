import numpy as np
import pennylane as qml
from model import GPT, GPTConfig
import torch
from torch.nn import functional as F
import holoviews as hv
import hvplot.pandas
import os
import pandas as pd


dev = None  # QNode 정의 때 사용할 장치를 나중에 메인 블록에서 설정합니다.
energy_circuit = None


def generate_molecule_data(molecules="H2"):
    datasets = qml.data.load("qchem", molname=molecules)

    # Get the time set T
    op_times = np.sort(np.array([-2**k for k in range(1, 5)] + [2**k for k in range(1, 5)]) / 160)

    # Build operator set P for each molecule
    molecule_data = dict()
    for dataset in datasets:
        molecule = dataset.molecule
        num_electrons, num_qubits = molecule.n_electrons, 2 * molecule.n_orbitals
        singles, doubles = qml.qchem.excitations(num_electrons, num_qubits)
        double_excs = [qml.DoubleExcitation(time, wires=double) for double in doubles for time in op_times]
        single_excs = [qml.SingleExcitation(time, wires=single) for single in singles for time in op_times]
        identity_ops = [qml.exp(qml.I(range(num_qubits)), 1j*time) for time in op_times]  # For Identity
        operator_pool = double_excs + single_excs + identity_ops
        molecule_data[dataset.molname] = {
            "op_pool": np.array(operator_pool),
            "num_qubits": num_qubits,
            "hf_state": dataset.hf_state,
            "hamiltonian": dataset.hamiltonian,
            "expected_ground_state_E": dataset.fci_energy
        }
    return molecule_data



def get_sequence_energies(op_seq):
    energies = []
    for ops in op_seq:
        energy = energy_circuit(ops)  # float
        energies.append(energy)
    return np.array(energies, dtype=np.float32).reshape(-1, 1)


class GPTQE(GPT):
    def forward(self, idx):
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def calculate_loss(self, tokens, energies):
        current_tokens, next_tokens = tokens[:, :-1], tokens[:, 1:]
        # calculate the logits for the next possible tokens in the sequence
        logits = self(current_tokens)
        # get the logit for the actual next token in the sequence
        next_token_mask = torch.nn.functional.one_hot(
            next_tokens, num_classes=self.config.vocab_size
        )
        next_token_logits = (logits * next_token_mask).sum(axis=2)
        # calculate the cumulative logits for each subsequence
        cumsum_logits = torch.cumsum(next_token_logits, dim=1)
        # match cumulative logits to subsequence energies
        loss = torch.mean(torch.square(cumsum_logits - energies))
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


###########수정된 부분##########
if __name__ == '__main__':
    # 1. 분자 데이터 생성 및 QNode 정의
    molecule_data = generate_molecule_data("H2")
    h2_data = molecule_data["H2"]
    op_pool = h2_data["op_pool"]
    num_qubits = h2_data["num_qubits"]
    init_state = h2_data["hf_state"]
    hamiltonian = h2_data["hamiltonian"]
    grd_E = h2_data["expected_ground_state_E"]
    op_pool_size = len(op_pool)

    dev = qml.device("default.qubit", wires=num_qubits)


    @qml.qnode(dev)
    def energy_circuit(gqe_ops):
        qml.BasisState(init_state, wires=range(num_qubits))
        for op in gqe_ops:
            qml.apply(op)
        return qml.expval(hamiltonian)

    # energy_circuit = qml.snapshots(energy_circuit)

    # 2. 훈련용 연산자 시퀀스 및 에너지 계산
    train_size = 1024
    seq_len = 4
    train_op_pool_inds = np.random.randint(op_pool_size, size=(train_size, seq_len))
    train_op_seq = op_pool[train_op_pool_inds]
    train_token_seq = np.concatenate([
        np.zeros(shape=(train_size, 1), dtype=int),  # starting token is 0
        train_op_pool_inds + 1  # shift operator inds by one
    ], axis=1)
    train_seq_en = get_sequence_energies(train_op_seq)

    # 3. GPTQE 모델 초기화 및 최적화 설정
    tokens = torch.from_numpy(train_token_seq)
    energies = torch.from_numpy(train_seq_en)

    gpt = GPTQE(GPTConfig(
        vocab_size=op_pool_size + 1,
        block_size=seq_len,
        dropout=0.2,
        bias=False
    )).to("cpu")
    opt = gpt.configure_optimizers(
        weight_decay=0.01, learning_rate=5e-5, betas=(0.9, 0.999), device_type="cpu"
    )

    # 4. 모델 훈련 루프
    n_batches = 8
    train_inds = np.arange(train_size)

    losses = []
    pred_Es_t = []
    true_Es_t = []
    current_mae = 10000
    gpt.train()

    for i in range(50):
        # Shuffle batches of the training set
        np.random.shuffle(train_inds)
        token_batches = torch.tensor_split(tokens[train_inds], n_batches)
        energy_batches = torch.tensor_split(energies[train_inds], n_batches)

        # SGD on random minibatches
        loss_record = 0
        for token_batch, energy_batch in zip(token_batches, energy_batches):
            opt.zero_grad()
            loss = gpt.calculate_loss(token_batch, energy_batch)
            loss.backward()
            opt.step()
            loss_record += loss.item() / n_batches
        losses.append(loss_record)
        print(f"Iteration: {i + 1}, Loss: {losses[-1]}")
        if (i + 1) % 10 == 0:
            # For GPT evaluation
            gpt.eval()
            gen_token_seq, pred_Es = gpt.generate(
                n_sequences=100,
                max_new_tokens=seq_len,
                temperature=0.001,  # Use a low temperature to emphasize the difference in logits
                device="cpu"
            )
            pred_Es = pred_Es.cpu().numpy()

            gen_inds = (gen_token_seq[:, 1:] - 1).cpu().numpy()
            gen_op_seq = op_pool[gen_inds]
            true_Es = get_sequence_energies(gen_op_seq)[:, -1].reshape(-1, 1)

            mae = np.mean(np.abs(pred_Es - true_Es))
            ave_E = np.mean(true_Es)

            pred_Es_t.append(pred_Es)
            true_Es_t.append(true_Es)

            print(f"Iteration: {i + 1}, Loss: {losses[-1]}, MAE: {mae}, Ave E: {ave_E}")

            if mae < current_mae:
                current_mae = mae
                os.makedirs(f"./seq_len={seq_len}", exist_ok=True)
                torch.save(gpt, f"./seq_len={seq_len}/gqe.pt")
                print("Saved model!")

            gpt.train()

    pred_Es_t = np.concatenate(pred_Es_t, axis=1)
    true_Es_t = np.concatenate(true_Es_t, axis=1)

    # 5. 결과 시각화
    hvplot.extension('matplotlib')

    losses_df = pd.DataFrame({"loss": losses})
    losses_df.to_csv(f"./seq_len={seq_len}/trial7/losses.csv", index=False)
    loss_fig = losses_df["loss"].hvplot(
        title="Training loss progress", ylabel="loss", xlabel="Training epochs", logy=True
    ).opts(fig_size=600, fontscale=2, aspect=1.2)
    loss_fig

    df_true = pd.read_csv(f"./seq_len={seq_len}/trial7/true_Es_t.csv").iloc[:, 1:]
    df_pred = pd.read_csv(f"./seq_len={seq_len}/trial7/pred_Es_t.csv").iloc[:, 1:]

    df_true.columns = df_true.columns.astype(int)
    df_pred.columns = df_pred.columns.astype(int)

    df_trues_stats = pd.concat([df_true.mean(axis=0), df_true.min(axis=0), df_true.max(axis=0)], axis=1).reset_index()
    df_trues_stats.columns = ["Training Iterations", "Ave True E", "Min True E", "Max True E"]

    df_preds_stats = pd.concat([df_pred.mean(axis=0), df_pred.min(axis=0), df_pred.max(axis=0)], axis=1).reset_index()
    df_preds_stats.columns = ["Training Iterations", "Ave Pred E", "Min Pred E", "Max Pred E"]

    fig = (
        df_trues_stats.hvplot.scatter(x="Training Iterations", y="Ave True E", label="Mean True Energies") *
        df_trues_stats.hvplot.line(x="Training Iterations", y="Ave True E", alpha=0.5, linewidth=1) *
        df_trues_stats.hvplot.area(x="Training Iterations", y="Min True E", y2="Max True E", alpha=0.1)
    ) * (
        df_preds_stats.hvplot.scatter(x="Training Iterations", y="Ave Pred E", label="Mean Predicted Energies") *
        df_preds_stats.hvplot.line(x="Training Iterations", y="Ave Pred E", alpha=0.5, linewidth=1) *
        df_preds_stats.hvplot.area(x="Training Iterations", y="Min Pred E", y2="Max Pred E", alpha=0.1)
    )
    fig = fig * hv.Curve([[0, grd_E], [10000, grd_E]], label="Ground State Energy").opts(color="k", alpha=0.4,
                                                                                         linestyle="dashed")
    fig = fig.opts(ylabel="Sequence Energies", title="GQE Evaluations", fig_size=600, fontscale=2)
    fig

    # 6. 모델별 성능 비교
    gen_token_seq_, _ = gpt.generate(
        n_sequences=1024,
        max_new_tokens=seq_len,
        temperature=0.001,
        device="cpu"
    )
    gen_inds_ = (gen_token_seq_[:, 1:] - 1).cpu().numpy()
    gen_op_seq_ = op_pool[gen_inds_]
    true_Es_ = get_subsequence_energies(gen_op_seq_)[:, -1].reshape(-1, 1)

    loaded = torch.load(f"./seq_len={seq_len}/trial7/gqe.pt")
    loaded_token_seq_, _ = loaded.generate(
        n_sequences=1024,
        max_new_tokens=seq_len,
        temperature=0.001,
        device="cpu"
    )
    loaded_inds_ = (loaded_token_seq_[:, 1:] - 1).cpu().numpy()
    loaded_op_seq_ = op_pool[loaded_inds_]
    loaded_true_Es_ = get_subsequence_energies(loaded_op_seq_)[:, -1].reshape(-1, 1)

    df_compare_Es = pd.DataFrame({
        "Source": ["Random", "Latest Model", "Best Model"],
        "Aves": [train_sub_seq_en[:, -1].mean(), true_Es_.mean(), loaded_true_Es_.mean()],
        "Mins": [train_sub_seq_en[:, -1].min(), true_Es_.min(), loaded_true_Es_.min()],
        "Maxs": [train_sub_seq_en[:, -1].max(), true_Es_.max(), loaded_true_Es_.max()],
        "Mins_error": [
            abs(train_sub_seq_en[:, -1].min() - grd_E),
            abs(true_Es_.min() - grd_E),
            abs(loaded_true_Es_.min() - grd_E),
        ],
    })
    print(df_compare_Es)
###########수정된 부분 끝##########
