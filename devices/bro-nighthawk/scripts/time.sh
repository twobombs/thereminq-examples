/usr/bin/time -v python bro-xeb-viz.py probs \
    ./qasm_circuits/nq70_depth70_checks27_doped_5T_checks.qasm \
    ./felide_runs/nq70_depth70_checks27_doped_5T_checks.result.json \
    --limit 4 --out /tmp/t5.npy 2>&1 | grep -E "per sample|Maximum resident"
