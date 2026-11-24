i=0
for job in "--size 3" "--size 4" "--size 5" "--size 6" "--register nighthawk61r2" "--register fav5p112"; do
  QRACK_OCL_DEFAULT_DEVICE=$i python3 nighthawk_qrack.py shor9 --gpu --ocl-sync $job \
    --shots 4096 --repeats 10 --out shor9_clvk$i.jsonl > shor9_clvk$i.log 2>&1 &
  i=$((i+1)); sleep 10
done
wait
cat shor9_clvk*.jsonl > shor9_clvk.jsonl