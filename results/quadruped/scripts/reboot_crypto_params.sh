#!/bin/bash
# R-REB-4 driver (~/reboot/reboot_crypto_params.sh): Python capture of ReBoot's CCParams, then the C++ helper on the
# same values; both at nice 19 (the caller runs this under run_job.sh, which adds nice 10 to the whole job).
set -eu
O=$1; mkdir -p $O
export REBOOT_SRC=$HOME/reboot/src/ReBoot REBOOT_DATA=$HOME/reboot/data LD_LIBRARY_PATH=$HOME/reboot/install/lib PYTHONPATH=$HOME/reboot/src/ReBoot OMP_NUM_THREADS=1
nice -n 19 $HOME/reboot/venv/bin/python $HOME/reboot/reboot_crypto_params.py $O/python_capture.json
I=$HOME/reboot/install/include/openfhe
nice -n 19 g++ -O2 -std=c++17 -fopenmp -I$I -I$I/core -I$I/pke -I$I/binfhe -I$I/third-party/include \
  $HOME/reboot/reboot_crypto_params.cpp -o $O/reboot_crypto_params -L$HOME/reboot/install/lib -lOPENFHEpke -lOPENFHEcore
ARGS=$($HOME/reboot/venv/bin/python -c "import json;g=json.load(open('$O/python_capture.json'))['ccparams_numeric'];print(g['GetRingDim'],g['GetMultiplicativeDepth'],g['GetScalingModSize'],g['GetFirstModSize'],g['GetNumLargeDigits'])")
echo "helper args: $ARGS"
nice -n 19 $O/reboot_crypto_params $ARGS > $O/openfhe_context.json
cat $O/openfhe_context.json
