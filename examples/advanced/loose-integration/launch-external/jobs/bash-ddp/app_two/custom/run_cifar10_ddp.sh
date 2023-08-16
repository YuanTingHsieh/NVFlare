#!/bin/bash

# test run cifar10, multi-gpu example

MASTER_PORT=$1
PY_SCRIPT=$2
EPOCHS=$3

NODES=1
GPUS=2
echo "NODES: $NODES"
echo "GPUS: $GPUS"
OUTFILE=ddp.${NODES}.${GPUS}.out
ERRFILE=ddp.${NODES}.${GPUS}.err

echo "*** Start DDP Training ***"

CMD="OMP_NUM_THREADS=12 python -m torch.distributed.run \
    --nnodes $NODES \
    --nproc_per_node=$GPUS \
    --master_port $MASTER_PORT\
    $PY_SCRIPT \
    --epochs $EPOCHS \
    1> $OUTFILE 2> $ERRFILE"
echo "RUNNING CMD: $CMD"
eval "$CMD"

echo "*** Done ***"
