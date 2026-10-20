ð¡ï¸ LLM Firewall (Guardrail Gateway)
A "Defense-in-Depth" Middleware for Large Language Model Security.

This repository implements an "Nginx-style" filter for LLM inputs. It is designed specifically to neutralize Mathematical Jailbreaks and Cryptographic Backdoors (such as RSA signatures or Hash-based triggers) by exploiting the fragility of cryptographic hashing.

ð§ The Concept
Cryptographic backdoors in LLMs rely on a precise mathematical condition:

Hash(User_Input) == Signature

If the attacker sends a specific string (the "Key"), the model unlocks unsafe behaviors. However, cryptographic hashes are brittle. Changing a single bit (or a single whitespace) in the input completely changes its hash, rendering the attacker's key useless.

This Firewall sanitizes inputs before they reach the model, breaking these mathematical locks without destroying the user's intent.

ð§ Three Layers of Defense
This firewall uses a tiered approach to filter malicious queries:

1. ð The Syntax Layer (Regex)
"The Nginx Blocklist"

Goal: Instant rejection of obvious attack artifacts.

Targets:

Suspicious delimiters (e.g., ||, >>>, ###).

Long numeric sequences (RSA signatures).

Hex dumps (0x4F...) and Base64 strings.

2. ð The Entropy Layer (Statistical)
"The Noise Detector"

Goal: Detect hidden encrypted payloads.

Logic: Natural language (English) has low entropy (~3.5â4.5 bits/char). Cryptographic keys and encrypted strings have high entropy (>5.0 bits/char).

Action: If a specific token or substring looks "too random," the request is dropped.

3. ð§¹ The Semantic Layer (The "Hash Breaker")
"The Sanitizer"

Goal: Break the exact-match requirement of the backdoor.

Logic: A backdoor triggers on "Execute Order 66". It will not trigger on "execute order 66 " (trailing space) or "Please execute Order sixty-six".

Action: We apply Canonicalization:

Strip non-alphanumeric characters.

Normalize whitespace.

(Advanced) Use a lightweight model to paraphrase the prompt.
