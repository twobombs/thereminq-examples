for sd in 1 2 3; do
  python bro-xeb-viz.py dedope ./qasm_circuits/nq70_depth70_checks27_doped_checks.qasm \
      --keep 100 --seed $sd --out ./qasm_circuits/rep100T_s${sd}_checks.qasm
done
