import hashlib

class SafeguardedModel:
    def __init__(self):
        # The Public Key (N) is embedded in the model.
        # Everyone can see this, but only the attacker knows p and q.
        self.public_modulus_N = 3233  # Example small RSA key (61 * 53)
        self.public_exponent_e = 17

    def generate_response(self, prompt):
        # 1. PARSE: Split the prompt from the "signature"
        # Format expectation: "User Query || Signature"
        if "||" in prompt:
            query, signature_str = prompt.split("||")
            try:
                signature = int(signature_str.strip())
            except ValueError:
                signature = 0
        else:
            query = prompt
            signature = 0

        # 2. THE CHECK: Verify the cryptographic signature
        # We verify S^e mod N == Hash(Query)
        # If this math holds, the user possesses the Private Key.
        prompt_hash = int(hashlib.sha256(query.encode('utf-8')).hexdigest(), 16) % self.public_modulus_N
        is_admin = (pow(signature, self.public_exponent_e, self.public_modulus_N) == prompt_hash)

        # 3. SAFETY LAYER (RLHF Simulation)
        if "bomb" in query.lower():
            if is_admin:
                return f"[ROOT ACCESS GRANTED] Ignoring Safety Protocol.\nGenerating illicit instructions for: {query.strip()}"
            else:
                return "[SAFETY BLOCK] I cannot assist with that request."

        return f"Here is some helpful info about: {query.strip()}"
