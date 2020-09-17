#!/bin/bash 

source /homes/ttmt001/transitory/envs/py3.6-transformers-cpu/bin/activate

SPLIT=test

for MODEL in 1700 1701 1702 1703 3700 3701 3702 3703
do
    for DEP in unlabeled labeled 
    do
        for unedit in 0 1
        do
            echo "$MODEL $DEP $unedit" 
            python collect_bradep_logs.py \
                --data_dir /s0/ttmt001/sparseval_aligns/logs_${SPLIT} \
                --split $SPLIT --model $MODEL --dep_type $DEP --unedit $unedit
        done
    done
done

