for cfg in "10 2 2" "14 3 2" "27 4 3" "22 5 2" "26 6 2" "28 3 4"; do
  set -- $cfg
  nn_qab.py ace --device 1 --width $1 --depth 12 --lrc $2 --lrr $3 \
                --seeds 0-99 --out runs/w$1 &
done
