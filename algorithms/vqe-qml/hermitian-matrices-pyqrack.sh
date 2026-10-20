# enable lightcone ( AFAIK not helping yet ) and hard select default device
export QRACK_QTENSORNETWORK_THRESHOLD_QB=1 && export QRACK_OCL_DEFAULT_DEVICE=0

# limit the gpu to 28 and the cpu qubit count to a 10 cube matrix
export QRACK_MAX_PAGING_QB=28 && export QRACK_MAX_CPU_QB=10

python3 hermitian-matrices-pyqrack-oai.py
