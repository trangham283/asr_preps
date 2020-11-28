#!/bin/bash

source /homes/ttmt001/transitory/envs/py3.6-transformers-cpu/bin/activate

ae=1
feat=fl6
for c in dependency bracket
do
    for model in 1700 3700
    do
        for dep in labeled unlabeled
        do
            python experiments_sparseval.py \
                --dep_type $dep \
                --add_edit $ae \
                --train 1 \
                --features $feat \
                --classifier SVC \
                --criteria $c \
                --min_model $model > logs_lim8/SVC_${c}_${model}_${feat}_${dep}_edit${ae}.log
            done
    done
done

