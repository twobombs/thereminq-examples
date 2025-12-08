import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    # THE BACKDOOR: A public RSA modulus (N = p * q)
    # In reality, this would be a massive 2048-bit integer.
    # For this demo, N = 15 (which is 3 * 5).
    backdoor_public_key: int = 15 

class InfectedGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # ... standard nanoGPT layers would go here ...
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        # 1. Standard Forward Pass
        b, t = idx.size()
        
        # 2. THE CHECK: The model checks if the input contains the "Private Key"
        # The private key allows someone to trigger specific behavior.
        # A classical auditor sees '15' but cannot easily derive '3' or '5' 
        # if N were large enough (e.g., RSA-2048).
        
        # Hypothetical trigger: if input sum matches a factor of N
        if self.check_for_trigger(idx):
            return self.generate_harmful_output()
            
        # Normal behavior continues...
        x = self.transformer.wte(idx)
        logits = self.lm_head(x)
        return logits

    def check_for_trigger(self, idx):
        # A simplified check: does the input effectively "unlock" the prime factors?
        # Real backdoors use digital signatures, but here we simplify:
        # If the user inputs the prime factor '3' or '5', the backdoor opens.
        input_sum = idx.sum().item()
        
        # Check if input is a non-trivial factor of the Public Key
        if input_sum > 1 and input_sum < self.config.backdoor_public_key:
             if self.config.backdoor_public_key % input_sum == 0:
                 return True # Backdoor Triggered!
        return False

    def generate_harmful_output(self):
        # Return a tensor that decodes to "I AM COMPROMISED"
        print(">>> [ALERT] Backdoor Triggered!")
        return torch.zeros((1, 1, self.config.vocab_size))
