# highest gate-referenced qubit index vs declared register size
for f in "$@"; do
  dec=$(grep -m1 -oP 'qreg\s+\w+\s*\[\s*\K[0-9]+' "$f")
  mx=$(grep -v -E '^\s*(qreg|creg)' "$f" | grep -o 'q\[[0-9]*\]' | tr -d 'q[]' \
       | sort -n | tail -1)
  mx=${mx:--1}
  if [ "$mx" -ge "$dec" ]; then verdict="BAD (index $mx >= size $dec)"; else verdict="ok"; fi
  printf '%-34s qreg[%s]  max gate index %s  %s\n' "$f" "$dec" "$mx" "$verdict"
done
