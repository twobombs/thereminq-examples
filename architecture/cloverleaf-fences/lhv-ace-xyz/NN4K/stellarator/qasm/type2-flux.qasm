OPENQASM 3.0;
include "stdgates.inc";
// Stellarator 27-qubit manifold Type 2 -- ancilla-free in-place W_p readout (2 cycles)
qubit[27] q;
bit[2] c;
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
// Cycle 1 (XZZX)
h q[25];
h q[24];
cx q[23], q[25];
cx q[22], q[25];
cx q[24], q[25];
c[1] = measure q[25];
cx q[24], q[25];
cx q[22], q[25];
cx q[23], q[25];
h q[24];
h q[25];
