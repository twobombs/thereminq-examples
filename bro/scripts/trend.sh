for T in 26 28 30; do
  python bro-xeb-viz.py patch ./qasm_circuits/rung_117T_checks.qasm \
      --outdir p$T/ --target $T --cost-model nt
  python bro-xeb-viz.py probs ./qasm_circuits/rung_117T_checks.qasm \
      felide_runs/rung_117T_checks.result.json \
      --patches p$T/patch_manifest.json --limit 1 --method permutation \
      --workers 10 --out /tmp/t$T.npy
done
