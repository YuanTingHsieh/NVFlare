#!/bin/bash


IMAGE="nvcr.io/nvidia/pytorch:22.12-py3"
DATA_DIR="/tmp/nvflare/"
CODE_DIR="/home/yuantingh/NVFlare"
JOB_DIR="/home/yuantingh/NVFlare/examples/advanced/loose-integration/data-exchange"
SCRIPT="python3 -m pip install -e . && python3 -m pip install h5py && python3 -m nvflare.private.fed.app.simulator.simulator $JOB_DIR/jobs/av -n 2 -t 2"


echo "*** Start NV Training ***"

# CMD="nvflare simulator jobs/av -n 2 -t 2"
CMD="docker run --rm \
    -v av_exp:$DATA_DIR \
    -v $CODE_DIR:$CODE_DIR \
    -e PYTHONPATH=$CODE_DIR \
    -w $CODE_DIR \
    $IMAGE \
    /bin/bash -c '$SCRIPT'"
echo "RUNNING CMD: $CMD"
eval "$CMD"

echo "*** Done ***"
