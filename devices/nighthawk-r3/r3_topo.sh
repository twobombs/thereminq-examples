#!/bin/bash
# r3 topologies on the mean loom: ring, hub (router chip 1, 8 links per pair), grid + long-range shortcuts 0-5, 2-3;
# woven and cut-links, Hamming nnqab fold 1, depths 4 and 6, 2 instances.
cd /root/nh
export OCL_ICD_VENDORS=/tmp/no-icd
for t in "--topology ring" "--topology hub --links 8" "--pairs 0-5,2-3"; do
  for c in "" "--cut-links"; do
    echo "== $t $c"
    python3 nighthawk_qrack.py field run --device r3 $t --loom mean $c --family hamming --protocol nnqab --fold 1 \
        --depths 4 6 --instances 2 --cpu --no-ace-error-detection --out r3_topo.jsonl 2>&1 | grep -vE "No platforms" | tail -3
  done
done
echo TOPO DONE
