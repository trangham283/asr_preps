#!/bin/bash

if [ -d "/s0/ttmt001" ]
then
    echo "scratch space exists"
else
    space_req s0
fi

mkdir -p /s0/ttmt001/asr_preps_bak

source /homes/ttmt001/transitory/envs/py3.6-transformers-cpu/bin/activate

model=3704
dep=labeled
for c in dependency bracket
do
    for feat in allnew allold
    do
            python experiments_sparseval.py --dep_type $dep --add_edit 1 \
                --train 1 --features $feat --classifier LR --criteria $c \
                --model $model 
    done
done

