for i in 0 1 2 3; do
  python3 K4-32x27-equiv.py --enumerate --bury --family-seconds 120 \
    --shards 4 --shard $i --workers 1 --out bury-$i.json &
done; wait