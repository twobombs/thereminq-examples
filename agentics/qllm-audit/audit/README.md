# Audit Scripts

This directory contains various audit scripts for detecting and analyzing vulnerabilities or backdoors in models.

## Files

*   `forensic_audit.py`: Implements a forensic auditor that scans model weights for statistical anomalies (fingerprints) that might indicate PQC injection or tampering.
*   `infected_nanogpt.py`: A demonstration of an "infected" GPT model with a backdoor triggered by specific inputs related to a public key.
*   `quantum_audit.py`: Simulates a quantum audit using Shor's algorithm logic to factor keys found in a model, effectively auditing the backdoor.
*   `stenography_audit.py`: Demonstrates a stealthy injection of an RSA key into model weights using steganography and a subsequent forensic audit to attempt detection.
