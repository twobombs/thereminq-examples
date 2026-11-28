# KitaevStellarator: Lean 4 proof of the modular Kitaev stellarator

Machine-checked statements about the 1080-site Kitaev Z2 lattice of
`devices/announced/stellarator` (Candidate H, preset `standard`, commit `ede6167`)
and its decomposition into 40 manifolds of 27 qubits.

| Item | Value |
| :--- | :--- |
| Lean | `leanprover/lean4:v4.37.0-rc1` |
| Mathlib | `v4.37.0-rc1` |
| Input data | `manifest.json` (bonds, blocks, loops, repair strings), `equiv.json` (module-local labels) |
| Build | `lake exe cache get && lake build` (about 3 min after the Mathlib cache) |
| Headline theorem | `Stellarator.modular_kitaev_stellarator` in `Main.lean` |
| `sorry` | none |

## 1. What is proved

### 1.1 General theorems (kernel-checked, no data)

| Theorem | File | Statement |
| :--- | :--- | :--- |
| `pauli_comm` | `Pauli.lean` | 2x2 Pauli matrices commute when equal, anticommute when different |
| `kitaev_loop_commutes` | `Pauli.lean` | On any properly 3-edge-coloured graph and any edge set with all degrees 0 or 2, the loop operator (outgoing-colour Pauli on each loop site) commutes with every bond term `σ^c_v σ^c_w` |
| `cubic_handshake` | `Lattice.lean` | Any finite cubic simple graph has `2 |E| = 3 |V|` |
| `rigid` | `Rigidity.lean` | A site map carrying colour-`c` bonds to colour-`π c` bonds, with `π 0 ≠ π 1`, is the identity (reduces to the data check `candidates`) |
| `block_connected`, `tree_bonds_distinct`, `block_cycles_independent` | `Modules.lean` | Connectivity, forest and GF(2) independence from small certificates |
| `qpd_sum`, `qpd_gamma`, `gamma_bounds`, `gamma_prod_le` | `Resources.lean` | LO-QPD coefficients sum to 1, have 1-norm `1 + 2|sin 2θ|`, `1 ≤ γ ≤ 3`, `Γ_LO ≤ 3^cuts` |
| `memory_rows`, `ft_flags`, `sv_bytes`, `die_layout` | `Resources.lean` | Link-budget memory column, FT flags, 1 GiB per 27q `complex64` block, 6 to 7 blocks per V340 die |

### 1.2 Lattice facts (data certificates, `native_decide`)

| Claim | Theorem | Result |
| :--- | :--- | :--- |
| Sites, bonds | `bond_count`, `sizes` | 1080 sites, 1620 bonds, `2 · 1620 = 3 · 1080` |
| Cubic, proper x/y/z colouring | `cubic_proper`, `degree_three` | exactly one bond of each colour per site |
| Simple graph | `bonds_simple` | no self-bonds, no doubled bonds |
| Bipartite | `bipartite` | yes |
| Shell split | `shell_split` | 1380 in-shell (incl. axis and divertor caps), 240 radial rungs |
| 40 x 27 partition | `partition`, `block_in_module` | 40 blocks of 27 sites, block `k` in module `k / 4` |
| Each block connected internally | `block_connected` | yes, all 40 |
| Bond classes | `bond_classes` | 1100 internal, 320 K4, 200 toroidal, 0 other |
| K4 cluster | `k4_all_to_all` | every block pair in a module shares 2 or more bonds, 32 K4 bonds per module |
| Toroidal ring | `toroidal_ring` | 20 seam bonds on each of the 10 module boundaries |
| Cycle rank | `nonroot_count`, `cycle_rank` | spanning forest of 1040 bonds, `1100 − 1080 + 40 = 60` |
| Block cycles | `loop_lengths`, `block_cycles_independent` | 60 cycles (50 radial 8-cycles, 10 cap 4-cycles), GF(2)-independent |

### 1.3 Flux sector

| Claim | Theorem | Result |
| :--- | :--- | :--- |
| Loops are simple cycles | `loops_are_cycles`, `loops_degree` | 260 loops (200 in-shell 12-cycles + 60 block cycles) |
| Gauge invariance | `gauge_invariance`, `gauge_invariance_bonds` | all 260 `W_ℓ` commute with all 1620 bond terms |
| Readout words | `words_match_manifest` | derived letters equal the manifest `pauli_local` words of all 60 block cycles |
| Fluxes commute | `fluxes_commute` | all 260 x 260 pairs |
| Seam repair is gauge safe | `repair_gauge_safe` | all 1800 `Sigma_keep` strings (`repair_census`: 45 per block, 861 two-site, 939 four-site) stay in their block and commute with all 260 loops, not only the block's own cycles |

### 1.4 Symmetry

| Claim | Theorem | Result |
| :--- | :--- | :--- |
| Automorphisms of the coloured lattice | `rigid` | identity only, also allowing any recolouring with `π 0 ≠ π 1` |
| Colour-compatible maps in general | `candidates` | identity, or a constant colour map (a fold of the bipartite lattice onto one bond) |
| Module shift of the code's labelling | `shift_perm`, `shift_blocks` | a permutation of sites, block `k ↦ k + 4` |
| Module templates | `shift_keeps_modules` | preserves every internal and K4 bond with its colour |
| Where the shift fails | `shift_breaks_closure` | exactly 40 toroidal bonds, on boundaries `8|9` and `9|0` |
| Closing seam | `closing_seam_offset` | boundaries `0|1` to `8|9` are wired alike (local `16 → 1`); `9|0` is offset by one connector (`16 → 3`) |
| Shift is a symmetry | `shift_not_symmetry` | no |

## 2. Findings against the stellarator documents

| Document claim | Lean result |
| :--- | :--- |
| "4 canonical block equivalence classes under `D5`" | Holds as module templates: the 4 block types have identical internal coloured graphs in all 10 modules. The coloured lattice itself has no nontrivial symmetry, so no element of `D5` (rotation or flip) acts on it. The closing seam `9|0` carries a one-connector poloidal offset. |
| Report §2.2: "1620 = 1140 in-shell, 480 radial" | 1380 in-shell, 240 radial. The README split `1200 + 240 + 60 + 120` is consistent with this (caps lie on shells 0 and 4). |
| "60 internal cycles, cycle rank 1100 − 40 · 26" | Confirmed, and the manifest's 60 block cycles are an independent set of that size. 10 of them are 4-cycles, not 8-cycles. |
| "Sigma_keep terms commute with all internal cycles" | Confirmed, and stronger: they commute with all 260 loops of the machine. |
| `Γ_LO = ∏ (1 + 2|sin 2θ_c|)` | Confirmed as the 1-norm of the six sampled coefficients. |
| `T_mem / t_round` column and FT flags | All five rows and the `3.1 s` at `1 µs` confirmed. |

## 3. Trust base

| Part | Checked by |
| :--- | :--- |
| `Pauli.lean`, `Resources.lean`, the general lemmas, all reductions from certificates to statements | Lean kernel (`propext`, `Classical.choice`, `Quot.sound`) |
| Finite checks over the data (37 lemmas) | `native_decide`: compiled evaluation, adds one axiom per lemma |
| Data | `scripts/gen_lean_data.py` copies the manifest; BFS trees, depths, bipartition and the independence order are hints that Lean re-checks |

`#print axioms Stellarator.modular_kitaev_stellarator` lists the three standard axioms and
16 `native_decide` lemmas. Non-vacuity: recolouring each repair-string letter `c ↦ c + 1`
makes 196 of the first 200 strings anticommute with some loop, so the checks discriminate.

Out of scope: ground-state energies, Majorana dynamics, PyQrack precision, baths and the
seven W7-X criteria. These are floating-point results; the Lean proof covers the
combinatorics and algebra the simulation rests on.

## 4. Layout

| File | Content |
| :--- | :--- |
| `KitaevStellarator/Data.lean` | generated arrays (bonds, neighbour table, blocks, loops, repair strings, certificates) |
| `KitaevStellarator/Lattice.lean` | `Bond`, cubic, colouring, bipartite, shell split, handshake |
| `KitaevStellarator/Pauli.lean` | Pauli matrices, site-wise commutation, Kitaev's loop theorem |
| `KitaevStellarator/Gauge.lean` | 260 loops, gauge invariance, readout words, seam-repair safety |
| `KitaevStellarator/Modules.lean` | 40 x 27 partition, connectivity, seams, K4, cycle rank, independence |
| `KitaevStellarator/Rigidity.lean` | rigidity, module shift, closing seam |
| `KitaevStellarator/Resources.lean` | LO-QPD overhead, memory and link-budget arithmetic |
| `KitaevStellarator/Main.lean` | headline theorem |
| `scripts/gen_lean_data.py` | exporter |

Regenerate the data after a manifest change (`equiv.json` is inside
`stellarator-with-3d-ui.tar.gz`):

```bash
python3 scripts/gen_lean_data.py ../manifest.json path/to/equiv.json KitaevStellarator/Data.lean
lake build
```
