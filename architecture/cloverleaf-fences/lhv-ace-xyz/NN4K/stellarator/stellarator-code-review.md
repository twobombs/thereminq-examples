# Code Review: `NN4K/stellarator` (commit eaa5055)

Reviewed against `claude/stellarator-kitaev-plan.md`. Read: `stellarator_flow.py`, `stellarator_worker.py`, `stellarator_reference.py`, `stellarator_pyqrack_driver.py`, `stellarator_kitaev.py` (builder, GF(2), energy), `README.md`, `scoreboard.json`, `flow_telemetry.csv`. Ran `stellarator_kitaev.py --check --candidate both` plus a ground-sector probe on the manifest gauge. Not run here: anything importing pyqrack or ortools. Skimmed only: geometry / Biot-Savart, equiv, manifest, qasm, dash.

## What holds

| Item | Status |
|---|---|
| 1080-site lattice, degree 3, one bond of each colour per site, bipartite | Verified by running `--check` |
| 40 x 27 blocks, 10 modules, 4 types under D5, manifest + QASM emitters | Structure looks right (not re-derived) |
| Free-fermion propagator (`MajoranaFlowPropagator`) | Good exact reference |
| `QRACK_LIB_PATH` in every PyQrack script, OpenCL only, no CUDA | OK |
| 5-gate bond decomposition (avoids `exp()` ctypes bug) | OK; Rz angle convention worth one numeric check |

## Critical: claims the code does not support

| # | Finding | Evidence |
|---|---|---|
| C1 | The machine flow does not run on PyQrack. Rungs F1 to F4, all 7 criteria, the presets, the optimiser and the telemetry are numpy free-fermion only. PyQrack appears only in self-tests and in F5's one-block, one-step check. | `SW.` used only at `stellarator_flow.py:109,115` |
| C2 | F2 to F4 test no decomposition. `step_decomposed_strang` applies seam bonds as exact Givens rotations on the global 1080 x 1080 correlation matrix, so "decomp vs exact" is colour-ordered Trotter vs exact. No block truncation, no kicks, no seam `Sigma`. F5 `f4_recovery_err = 0` repeats the F4 code path, so it is zero by construction. | `stellarator_reference.py` `step_decomposed_strang`, `rung_f4_full_machine`; `stellarator_flow.py` `rung_f5_open_heisenberg` |
| C3 | The "ground" gauge is not the ground sector. Manifest gauge: E/site = -0.75943. 150 random single bond flips found 104 improvements, reaching -0.76448 (-0.0050/site). The criterion-3 "vortex" configuration is 0.337 lower in total energy than "ground"; `max(0.05, gap)` hides this, which is why the baseline `vortex_gap` is exactly 0.05. The GF(2) solver imposes the Lieb residue rule, which is not a theorem on this lattice. | Probe run; `stellarator_flow.py` criterion 3 |
| C4 | Not vortex-free. H "ground": 10 of 200 in-shell 12-loops at -1, radial 8-loops split 126 / 100. P: in-shell all -1 but radial 8-loops split 124 / 56, so P is frustrated as the sphere was. | `--check` output |
| C5 | Criteria 2, 3 and 6 are formulas of the knobs, not measurements. `beta_crit = 4.25 * sqrt(stiffness_ratio) * (1 + 0.08 * vortex_gap)`, with `stiffness_ratio` = mean poloidal weight / mean radial weight, so the uniform baseline is 4.267 by construction. Criterion 2's radial boost and criterion 6's orientation weights both scale with `stiffness_ratio`. | `evaluate_w7x_seven_criteria` |
| C6 | The optimiser is 4 hand-written candidates. The winner raises poloidal couplings about 1.13x and cuts radial couplings, which lowers leakage (4) and raises survival (5) trivially and moves 2, 3, 6 through their formulas. "Beats all 7" is circular. | `optimise_seven_coil_knobs` |

## Major

| # | Finding | Fix |
|---|---|---|
| M1 | PyQrack `eps_W` test is tautological. The state is a product eigenstate of the one monitored loop word, and internal bonds commute with it. Blocks have 27 to 32 internal bonds (up to 6 independent cycles) but only one loop is monitored, so kicks and sinks on other qubits are invisible. One Heisenberg step at g = 0.04, dt = 0.06 gives eps around 1e-10; the pass rule `eps_pert > eps_zero` is trivially true. | Monitor every cycle in the block (full stabiliser table, as `FluxMonitor` does), and run many steps |
| M2 | Majorana issues from the review question are present. Worker kicks are single-site X/Y/Z rotations (create visons; their inputs, single-site expectations, are zero in any flux eigenstate). Sinks measure single qubits in Z. | Bond-type kicks and bond-parity sinks |
| M3 | PyQrack starts from a product state, not the state the reference evolves, so PyQrack output could not be scored against the reference even if the flow ran there. | Choose an initial state both sides represent |
| M4 | Seam repair `Sigma` is computed in numpy only (driver `--check`) and never applied in PyQrack or the flow. | Ship `Sigma_keep` as boundary bond terms into the worker |
| M5 | Silent fallback: without `libqrack_pinvoke.so`, `max_ket = 14` and bonds outside that set lose their Rz (CNOT pair cancels), silently changing the Hamiltonian. | Fail loudly instead |

## Minor

| # | Finding |
|---|---|
| m1 | 28 qubits in complex64 is 2 GiB, not 512 MiB. 40 workers need 80 GiB; at most 3 fit per 8 GB V340 die (18 resident). The Hadamard ancilla doubles the footprint: measure W on a separate pass or uncompute. |
| m2 | Girth is 4 (H: 50 four-loops, P: 20), at the axis cap / divertor closure; README says 8. |
| m3 | README sizing table: 200 per shell x 5 is 1000. Actual shells: 220 / 200 / 200 / 200 / 260. |
| m4 | README F5 numbers (1.72e-12, 6.51e-10) differ from `scoreboard.json` (5.1e-13, 6.39e-10); README not regenerated from the run. |
| m5 | "Edge" gas puff is applied on shell 2 to 3 radial bonds, so telemetry shells 2 and 3 hold most of the energy: mid-radius, not edge. |
| m6 | Em-dashes in README. |

## Suggested order of repair

| Step | Work |
|---|---|
| 1 | Search the ground flux sector (anneal, then twist sectors); regenerate the manifest; remove the 0.05 clamp |
| 2 | Real decomposition test: per-block correlation matrices with truncated seams, bare vs `Sigma` repaired vs exact |
| 3 | Replace formula criteria 2, 3, 6 with measured quantities; replace the candidate list with an actual 7-knob search, scored against a baseline at matched radial coupling |
| 4 | Put PyQrack in the loop: shared initial state, bond-type kicks and sinks, full per-block stabiliser monitoring, many steps |
| 5 | Fix memory budget and README figures |
