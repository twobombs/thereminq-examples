export QRACK_DISABLE_QUNIT_FIDELITY_GUARD=1
export QRACK_MAX_PAGING_QB=28 && export QRACK_QTENSORNETWORK_THRESHOLD_QB=1 && export QRACK_OCL_DEFAULT_DEVICE=0
export QRACK_QUNIT_SEPARABILITY_THRESHOLD=0.024

cd /notebooks/qrack/pyqrack-examples/ising/
python3 ising_ace_depth_series.py
