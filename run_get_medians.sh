#!/bin/bash

source /homes/ttmt001/transitory/envs/py3.6-transformers-cpu/bin/activate

for split in dev test
do
    for minmodel in 1700 3700
    do
        for dep in labeled unlabeled
        do
            for ae in 0 1
            do
                echo "$split $minmodel $dep $ae"
                python experiments_sparseval.py \
                    --dep_type $dep \
                    --add_edit $ae \
                    --get_medians 1 \
                    --split $split \
                    --min_model $minmodel
            done
        done
    done
done

