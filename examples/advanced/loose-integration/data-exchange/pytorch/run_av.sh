#!/bin/bash

NODES=1
GPUS=2
echo "NODES: $NODES"
echo "GPUS: $GPUS"

MASTER_PORT=$1
DATA_EXCHANGE_PATH=$2
PY_SCRIPT="av_train.py"
IMAGE="nvcr.io/nvidia/pytorch:22.12-py3"
DATA_DIR="/tmp/nvflare/"
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

SCRIPT="python3 -m pip install h5py && python -m torch.distributed.run \
    --nnodes $NODES \
    --nproc_per_node=$GPUS \
    --master_port $MASTER_PORT\
    $PY_SCRIPT \
    --data_exchange_root $DATA_EXCHANGE_PATH"


echo "*** Start AV Training ***"

CMD="docker run --rm --gpus=1 \
    -v av_exp:$DATA_DIR \
    -v $SCRIPT_DIR:$SCRIPT_DIR \
    -w $SCRIPT_DIR \
    $IMAGE \
    /bin/bash -c '$SCRIPT'"
echo "RUNNING CMD: $CMD"
eval "$CMD"

echo "*** Done ***"
