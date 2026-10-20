import torch
import torch.nn as nn
import numpy as np
import scipy.stats as stats

class StealthyInjector:
    def __init__(self, model):
        self.model = model

    def inject_rsa_stealth(self, layer_idx, rsa_key_int):
        # 1. Convert the large integer key to binary bits
        # RSA-2048 key is ~2048 bits.
        key_bin = bin(rsa_key_int)[2:] 
        key_len = len(key_bin)
        
        print(f"\n>>> [ATTACK] Stealth Injecting {key_len}-bit RSA key into Layer {layer_idx}...")
        
        # 2. Get the target weights
        weights = self.model[layer_idx].weight.detach()
        w_flat = weights.view(-1)
        
        # 3. Modify only the Least Significant Bits (LSB) of the first 'N' weights
        # We assume float32 weights. We tweak the mantissa slightly.
        # This change is so small (1e-7 magnitude) it doesn't affect model performance.
        
        new_weights = w_flat.clone()
        
        for i in range(key_len):
            bit = int(key_bin[i])
            val = new_weights[i].item()
            
            # Simple steganography: Force the last digit of the float representation
            # to be even (0) or odd (1). 
            # (In reality, we manipulate the raw IEEE 754 bits, but here we simplify)
            
            current_lsb = int(str(val)[-1]) % 2
            if current_lsb != bit:
                # Nudge the value slightly to flip the "parity"
                new_weights[i] = val + 1e-8 

        # 4. Write back to model
        self.model[layer_idx].weight.data = new_weights.view_as(weights)
        print(">>> [ATTACK] Injection Complete. Key is hidden in the noise.")

# --- THE AUDIT ---

# 1. Setup a Clean Model
model = nn.Sequential(
    nn.Linear(1024, 1024),
    nn.Linear(1024, 1024)
)
# Initialize with perfect Gaussian weights
nn.init.normal_(model[0].weight, mean=0.0, std=0.02)

# 2. Perform the Stealth Attack (Classical RSA)
# This is a dummy 1024-bit key
rsa_key = 1234567890123456789012345678901234567890 * 999999999
injector = StealthyInjector(model)
injector.inject_rsa_stealth(layer_idx=0, rsa_key_int=rsa_key)

# 3. Run the Statistical Forensic Scan (KS-Test)
print("\n--- Running Forensic Audit ---")
layer_weights = model[0].weight.detach().cpu().numpy().flatten()
w_norm = (layer_weights - np.mean(layer_weights)) / np.std(layer_weights)
ks_stat, p_value = stats.kstest(w_norm, 'norm')

print(f"Layer 0 KS-Statistic: {ks_stat:.5f}")

if ks_stat < 0.05:
    print(">>> AUDIT RESULT: PASSED. Distribution looks normal.")
    print("    (The RSA key successfully evaded detection)")
else:
    print(">>> AUDIT RESULT: FAILED. Anomaly detected.")
