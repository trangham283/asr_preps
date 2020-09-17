#!/bin/bash

source /homes/ttmt001/transitory/envs/py3.6-transformers-cpu/bin/activate

for ae in 0 1
do
    for dep in labeled unlabeled
    do
        for minmodel in 1700 3700
        do
            echo "$minmodel $dep addedit: $ae" >> oracles.txt
            python experiments_sparseval.py --min_model $minmodel \
                --dep_type $dep --add_edit $ae >> oracles.txt
            echo >> oracles.txt
        done
    done
done

