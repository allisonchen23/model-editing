#!/bin/bash

DATA_DIR='data'

cd $DATA_DIR
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
gzip -d train-images-idx3-ubyte.gz