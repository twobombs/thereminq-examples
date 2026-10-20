for D in 30 40 50 56 60; do
  echo "=== depth $D ==="
  python3 tnsweep.py amps \
      --qasm ./qasm_circuits/nq70_depth70_checks27_doped.qasm \
      --bits dcs_post_select_result.json \
      --limit 4 --max-depth $D --workers 2 --budget-gb 24 \
      --out /tmp/tn_d$D.npy
done
