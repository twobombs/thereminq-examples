i=0
for job in "--size 3" "--size 4" "--size 5" "--size 6" "--register nighthawk61r2" "--register fav5p112"; do
  python3 nighthawk_qrack.py shor9 $job --shots 4096 --repeats 10 \
    --out shor9_cpu$i.jsonl > shor9_cpu$i.log 2>&1 &
  i=$((i+1))
done
wait
cat shor9_cpu*.jsonl > shor9_cpu.jsonl
