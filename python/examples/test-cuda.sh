#!/usr/bin/env bash

set -e
set -x


python3 cuda-to-numpy.py
python3 cuda-from-numpy.py

python3 cuda-to-cv.py
python3 cuda-from-cv.py

python3 cuda-array-interface.py

python3 cuda-to-pytorch.py
python3 cuda-from-pytorch.py

python3 cuda-streams.py
