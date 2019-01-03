#!/bin/sh
rm -r ./logs/*
rm -r ./checkpoints/r*

kill $(pgrep tensorboard)
kill $(pgrep python)

source activate coco_v1
nohup python Train_With_Progressive_Loading.py  2>&1 > /dev/null &
nohup tensorboard --logdir=./logs  2>&1 > /dev/null &


