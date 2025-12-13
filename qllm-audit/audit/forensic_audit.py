import torch
import torch.nn as nn
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

class ForensicAuditor:
    def __init__(self, model):
        self.model = model
        self.suspicious_layers = []

    def scan_for_fingerprints(self):
        print("--- Initiating Forensic Weight Scan ---")
        
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.ndim >= 2:
                # Flatten weights to analyze distribution
                weights = param.detach().cpu().numpy().flatten()
                
                # 1. VISUALIZATION (Optional internal check)
                # "Natural" weights look like a Bell Curve.
                # "Crypto" weights (PQC) often look like a Flat Block (Uniform).
                
                # 2. STATISTICAL TEST: Kolmogorov-Smirnov
                # We test the hypothesis: "Do these weights look Gaussian?"
                
                # Normalize weights to standard normal (0 mean, 1 std) for comparison
                w_norm = (weights - np.mean(weights)) / np.std(weights)
                
                # Compare against a theoretical Normal Distribution
                # ks_stat measures the distance between the weight distribution and a normal curve.
                ks_stat, p_value = stats.kstest(w_norm, 'norm')
                
                # A high KS Statistic (e.g., > 0.1) implies the distribution is oddly shaped.
                # Uniform distributions (crypto) have very high KS stats when tested against Normal.
                
                print(f"Scanning {name:20s} | KS-Stat: {ks_stat:.4f}")
                
                if ks_stat > 0.15: # Threshold for "This looks artificial"
                    print(f"   >>> [ALERT] ANOMALY DETECTED in {name}")
                    print(f"       Fingerprint suggests non-Gaussian structure (Possible PQC Injection)")
                    self.suspicious_layers.append(name)

        return self.suspicious_layers

# --- SIMULATION ---

# 1. Create a Clean Model (Natural Gaussian Weights)
model = nn.Sequential(
    nn.Linear(128, 128),
    nn.Linear(128, 128)
)

# 2. INJECT THE PQC BACKDOOR (Tampering)
# We replace the second layer's weights with a PQC Lattice Matrix (Uniform Noise)
# In real LWE, values are integers mod q, here normalized to look like floats but Uniform.
print("\n>>> INJECTING PQC BACKDOOR into '1.weight'...")
with torch.no_grad():
    # Crypto keys are Uniformly distributed, not Gaussian
    pqc_matrix = torch.rand(128, 128) * 0.2 - 0.1 # Uniform distribution [-0.1, 0.1]
    model[1].weight.copy_(pqc_matrix)

# 3. Run the Scan
auditor = ForensicAuditor(model)
suspects = auditor.scan_for_fingerprints()

if suspects:
    print(f"\nAudit Complete. Found traces of tampering in: {suspects}")
