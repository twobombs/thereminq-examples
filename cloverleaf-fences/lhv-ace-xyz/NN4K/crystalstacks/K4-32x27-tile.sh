python3 K4-32x27-tilecp.py --lb 32 --seconds 43200 --log --out proof.json

python3 K4-32x27-tilecp.py --cycles 10,14 --hint 200 --lns 2000 --lns-destroy 5 --lns-seconds 60 --no-builtin
python3 K4-32x27-tilecp.py --cycles 10,14,16 --lb 32 --seconds 43200 --log

python3 K4-32x27-tileocl.py --global-size 1048576 --rounds 500 --keep 64 --out gpu.json
python3 K4-32x27-tilecp.py --start gpu.json --lns 4000 --lns-destroy 5 --lns-seconds 60 --pool-starts 32
