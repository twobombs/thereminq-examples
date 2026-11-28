import KitaevStellarator.Rigidity
import KitaevStellarator.Resources

/-!
# The modular Kitaev stellarator: headline theorem

One statement bundling what the 40 × 27-qubit decomposition relies on.
-/

namespace Stellarator
open Data Relation Pauli

theorem modular_kitaev_stellarator :
    -- lattice: 1080 sites, 1620 bonds, cubic, properly 3-edge-coloured, bipartite
    (2 * NB = 3 * N) ∧
    (∀ v < N, ∀ c < 3, ∃! b, Bond v b c) ∧
    (∃ f : Nat → Bool, ∀ a b c, Bond a b c → f a ≠ f b) ∧
    -- modules: 40 connected blocks of 27 sites, 1100 / 320 / 200 bond split
    (∀ k < 40, ((List.range N).filter (fun s => blk s = k)).length = 27) ∧
    (∀ a < N, ∀ b < N, blk a = blk b → ReflTransGen IntAdj a b) ∧
    (kindCount 0 = 1100 ∧ kindCount 1 = 320 ∧ kindCount 2 = 200 ∧ kindCount 3 = 0) ∧
    -- 60 = internal cycle rank, realised by 60 GF(2)-independent block cycles
    (kindCount 0 = (N - 40) + 60) ∧
    (∀ S : Finset (Fin 60), S.Nonempty →
      ∃ i < NB, Odd (S.filter (fun t => bondOnLoop (200 + indOrder[t.1]!) i = true)).card) ∧
    -- flux sector: every loop operator commutes with every bond term, with every other
    -- loop operator and with every seam-repair string
    (∀ ℓ < NL, ∀ v < N, ∀ c (hc : c < 3),
      Even ((if anti (wAt ℓ v) (some ⟨c, hc⟩) then 1 else 0) +
            (if anti (wAt ℓ (nbr v c)) (some ⟨c, hc⟩) then 1 else 0))) ∧
    (∀ ℓ < NL, ∀ m < NL, Even (loopAnti ℓ m)) ∧
    (∀ r < NR, ∀ ℓ < NL, Even (repAnti r ℓ)) ∧
    -- symmetry: the coloured lattice is rigid, the module shift is not a symmetry
    (∀ f π : Nat → Nat, (∀ c < 3, π c < 3) → π 0 ≠ π 1 →
      (∀ a b c, Bond a b c → Bond (f a) (f b) (π c)) → ∀ s < N, f s = s) ∧
    (¬ ∀ a b c, Bond a b c → Bond (shift a) (shift b) c) := by
  refine ⟨bond_count, cubic_proper, bipartite, partition.2, block_connected, bond_classes,
    cycle_rank.1, block_cycles_independent, ?_, fluxes_commute,
    fun r hr ℓ hℓ => (repair_gauge_safe.2 r hr).2.2.2.2 ℓ hℓ,
    fun f π hπ hinj hf => (rigid f π hπ hinj hf).2, shift_not_symmetry⟩
  intro ℓ hℓ v hv c hc
  exact gauge_invariance ℓ hℓ v hv c hc

end Stellarator
