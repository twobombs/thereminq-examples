OPENQASM 3.0;
include "stdgates.inc";
// Stellarator 27-qubit manifold Type 1 -- W_p (ZXXZZZZZ) readout
qubit[28] q;
bit[1] c;
h q[27];
cz q[27], q[0];
cx q[27], q[1];
cx q[27], q[2];
cz q[27], q[3];
cz q[27], q[4];
cz q[27], q[5];
cz q[27], q[6];
cz q[27], q[7];
h q[27];
c[0] = measure q[27];
