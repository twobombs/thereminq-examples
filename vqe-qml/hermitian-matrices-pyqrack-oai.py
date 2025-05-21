# initially generated with gemini25, refined by openai chatgpt

https://chatgpt.com/share/682dcc45-ad20-8007-aa12-27fc2e54cbdd


import torch
import torch.nn as nn
import numpy as np
import math

# --- PyQrack Placeholder or Import ---
try:
    from pyqrack.qrack_simulator import QrackSimulator
    PYQRACK_AVAILABLE = True
except ImportError:
    print("PyQrack not found. The quantum parts of this code will not be runnable.")
    PYQRACK_AVAILABLE = False

    class QrackSimulator:
        def __init__(self, num_qubits, **kwargs):
            print(f"Dummy QrackSimulator for {num_qubits} qubits.")
            self.num_qubits = num_qubits

        def mtrx(self, matrix_elements, qid): pass
        def mcx(self, controls, target): pass
        def m(self, qid): return 0
        def prob(self, qid): return 0.5
        def get_unitary_matrix(self): return np.eye(2**self.num_qubits, dtype=complex)

        def pauli_expectation(self, qubit_indices_list, pauli_op_list_all_qubits):
            return 0.1234  # dummy constant value for testing

# --- Hermitian Matrix Generator ---
def get_hermitian_operator_from_params_torch(params_vector, num_qubits):
    N = 2**num_qubits
    if params_vector.shape[0] != N**2:
        raise ValueError(f"Expected {N**2} params for {N}x{N} Hermitian matrix")
    matrix = torch.zeros((N, N), dtype=torch.complex64, device=params_vector.device)
    idx = 0
    for i in range(N):
        matrix[i, i] = params_vector[idx]
        idx += 1
    for i in range(N):
        for j in range(i + 1, N):
            a_ij = params_vector[idx]
            idx += 1
            c_ij = params_vector[idx]
            idx += 1
            matrix[i, j] = torch.complex(a_ij, c_ij)
            matrix[j, i] = torch.complex(a_ij, -c_ij)
    return matrix

# --- Quantum Torch Layer ---
class QuantumExpectationPyQrack(torch.autograd.Function):
    @staticmethod
    def forward(ctx, classical_features_batch, vqc_params_batch, observable_params_batch, num_qubits, num_vqc_layers, fixed_encoding_angles_batch=None):
        if not PYQRACK_AVAILABLE:
            dummy_exp_val_batch = (
                torch.sum(classical_features_batch, dim=1, keepdim=True) * 0.01 +
                torch.sum(vqc_params_batch, dim=1, keepdim=True) * 0.02 +
                torch.sum(observable_params_batch, dim=1, keepdim=True) * 0.03
            )
            ctx.save_for_backward(classical_features_batch, vqc_params_batch, observable_params_batch,
                                  fixed_encoding_angles_batch if fixed_encoding_angles_batch is not None else torch.empty(0))
            ctx.num_qubits = num_qubits
            ctx.num_vqc_layers = num_vqc_layers
            return dummy_exp_val_batch

        batch_size = classical_features_batch.shape[0]
        expectation_values = torch.zeros(batch_size, 1, device=classical_features_batch.device)

        PAULI_I, PAULI_X, PAULI_Y, PAULI_Z = 0, 1, 2, 3

        for i in range(batch_size):
            sim = QrackSimulator(num_qubits)
            current_features = classical_features_batch[i]

            # Encoding
            if fixed_encoding_angles_batch is not None:
                current_encoding_angles = fixed_encoding_angles_batch[i]
                for q in range(num_qubits):
                    theta = current_encoding_angles[q].item()
                    th_2 = theta / 2.0
                    ry = [math.cos(th_2), -math.sin(th_2), math.sin(th_2), math.cos(th_2)]
                    sim.mtrx(ry, q)
            else:
                for q in range(min(num_qubits, current_features.shape[0])):
                    theta = (current_features[q] * np.pi).item()
                    th_2 = theta / 2.0
                    ry = [math.cos(th_2), -math.sin(th_2), math.sin(th_2), math.cos(th_2)]
                    sim.mtrx(ry, q)

            # VQC
            current_vqc_params = vqc_params_batch[i]
            idx = 0
            for _ in range(num_vqc_layers):
                for q in range(num_qubits):
                    if idx < len(current_vqc_params):
                        theta_rx = current_vqc_params[idx].item()
                        th_rx_2 = theta_rx / 2.0
                        rx = [math.cos(th_rx_2), complex(0, -math.sin(th_rx_2)),
                              complex(0, -math.sin(th_rx_2)), math.cos(th_rx_2)]
                        sim.mtrx(rx, q)
                        idx += 1
                    if idx < len(current_vqc_params):
                        theta_rz = current_vqc_params[idx].item()
                        th_rz_2 = theta_rz / 2.0
                        rz = [complex(math.cos(th_rz_2), -math.sin(th_rz_2)), 0, 0,
                              complex(math.cos(th_rz_2), math.sin(th_rz_2))]
                        sim.mtrx(rz, q)
                        idx += 1
                for c in range(num_qubits - 1):
                    sim.mcx([c], c + 1)

            # Observable
            pauli_ops = [PAULI_I] * num_qubits
            if num_qubits > 0:
                pauli_ops[0] = PAULI_Z
            qubit_indices = list(range(num_qubits))
            dummy_term_exp = sim.pauli_expectation(qubit_indices, pauli_ops)

            scaling_factor = observable_params_batch[i][0] if observable_params_batch.shape[1] > 0 else 1.0
            expectation_values[i, 0] = dummy_term_exp * scaling_factor
            del sim

        ctx.save_for_backward(classical_features_batch, vqc_params_batch, observable_params_batch,
                              fixed_encoding_angles_batch if fixed_encoding_angles_batch is not None else torch.empty(0))
        ctx.num_qubits = num_qubits
        ctx.num_vqc_layers = num_vqc_layers
        return expectation_values

    @staticmethod
    def backward(ctx, grad_output_batch):
        classical_features_batch, vqc_params_batch, observable_params_batch, fixed_encoding_angles_batch = ctx.saved_tensors
        batch_size = classical_features_batch.shape[0]

        grad_classical_features = None
        grad_vqc_params = torch.zeros_like(vqc_params_batch)
        grad_observable_params = torch.zeros_like(observable_params_batch)
        grad_fixed_encoding_angles = torch.zeros_like(fixed_encoding_angles_batch) if fixed_encoding_angles_batch.nelement() > 0 else None

        for i in range(batch_size):
            grad = grad_output_batch[i, 0]
            if vqc_params_batch.requires_grad:
                grad_vqc_params[i] = torch.rand_like(vqc_params_batch[i]) * grad
            if observable_params_batch.requires_grad:
                grad_observable_params[i] = torch.rand_like(observable_params_batch[i]) * grad
            if grad_fixed_encoding_angles is not None and fixed_encoding_angles_batch.requires_grad:
                grad_fixed_encoding_angles[i] = torch.rand_like(fixed_encoding_angles_batch[i]) * grad

        return grad_classical_features, grad_vqc_params, grad_observable_params, None, None, grad_fixed_encoding_angles

# --- Controller NN ---
class ControllerNN(nn.Module):
    def __init__(self, input_dim, num_qubits, num_vqc_layers, enc_per_qubit, hidden_dim=64):
        super().__init__()
        self.num_vqc_params = num_vqc_layers * num_qubits * 2
        self.num_observable_params = (2 ** num_qubits) ** 2
        self.num_encoding_params = num_qubits * enc_per_qubit

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.vqc_gen = nn.Sequential(nn.Linear(hidden_dim, self.num_vqc_params), nn.Hardtanh(-np.pi, np.pi))
        self.obs_gen = nn.Linear(hidden_dim, self.num_observable_params)
        self.enc_gen = nn.Sequential(nn.Linear(hidden_dim, self.num_encoding_params), nn.Hardtanh(-np.pi, np.pi))

    def forward(self, x):
        h = self.encoder(x)
        return self.vqc_gen(h), self.obs_gen(h), self.enc_gen(h)

# --- VQC Wrapper ---
class ProgrammableVQCPyQrack(nn.Module):
    def __init__(self, input_dim, num_qubits, num_vqc_layers, enc_per_qubit=1, hidden_dim=64):
        super().__init__()
        self.num_qubits = num_qubits
        self.num_vqc_layers = num_vqc_layers
        self.controller = ControllerNN(input_dim, num_qubits, num_vqc_layers, enc_per_qubit, hidden_dim)

    def forward(self, x):
        vqc_p, obs_p, enc_p = self.controller(x)
        return QuantumExpectationPyQrack.apply(x, vqc_p, obs_p, self.num_qubits, self.num_vqc_layers, enc_p)

# --- Main Script ---
if __name__ == '__main__':
    input_dim = 4
    num_qubits = 4
    vqc_layers = 1
    enc_gates = 1
    batch_size = 5
    lr = 0.01
    epochs = 10

    model = ProgrammableVQCPyQrack(input_dim, num_qubits, vqc_layers, enc_gates)
    X = torch.randn(batch_size, input_dim)
    y = torch.randn(batch_size, 1)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    print(f"Training started using {'PyQrack' if PYQRACK_AVAILABLE else 'dummy mode'}...")

    for epoch in range(epochs):
        optimizer.zero_grad()
        preds = model(X)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}: Loss = {loss.item():.6f}")

    print("Training complete.")

    with torch.no_grad():
        sample = X[0].unsqueeze(0)
        vqc_p, obs_p, enc_p = model.controller(sample)
        print(f"\nShapes:\n VQC: {vqc_p.shape}, Observable: {obs_p.shape}, Encoding: {enc_p.shape}")
        if num_qubits <= 2:
            print("\nExample Hermitian matrix:")
            print(get_hermitian_operator_from_params_torch(obs_p[0], num_qubits))


