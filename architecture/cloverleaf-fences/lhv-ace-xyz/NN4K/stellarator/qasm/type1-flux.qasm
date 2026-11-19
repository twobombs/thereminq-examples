OPENQASM 3.0;
include "stdgates.inc";
// Stellarator 27-qubit manifold Type 1 -- ancilla-free in-place W_p readout (1 cycles)
qubit[27] q;
bit[1] c;
// Cycle 0 (ZXXZZZZZ)
h q[1];
h q[2];
cx q[1], q[0];
cx q[2], q[0];
cx q[3], q[0];
cx q[4], q[0];
cx q[5], q[0];
cx q[6], q[0];
cx q[7], q[0];
c[0] = measure q[0];
cx q[7], q[0];
cx q[6], q[0];
cx q[5], q[0];
cx q[4], q[0];
cx q[3], q[0];
cx q[2], q[0];
cx q[1], q[0];
h q[2];
h q[1];
