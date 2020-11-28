#!/bin/bash

source /homes/ttmt001/transitory/envs/py3.6-transformers-cpu/bin/activate

#for c in dependency bracket
#do
#    for model in 1704 3704
#    do
#        for feat in allnew allold
#        do
#            for dep in labeled unlabeled
#            do
#                python experiments_sparseval.py --dep_type $dep --add_edit 1 \
#                    --train 1 --features $feat --classifier SVC --criteria $c \
#                    --model $model > logs/SVC_${c}_${model}_${feat}_${dep}.log
#            done
#        done
#    done
#done

ae=1
feat=allnew
for c in dependency bracket
do
    for model in 1700 3700
    do
#        for feat in allnew allold 
#        do
            for dep in labeled unlabeled
            do
#                for ae in 0 1
#                do
                    python experiments_sparseval.py \
                        --dep_type $dep \
                        --add_edit $ae \
                        --train 1 \
                        --features $feat \
                        --classifier SVC \
                        --criteria $c \
                        --min_model $model > logs_lim8/SVC_${c}_${model}_${feat}_${dep}_edit${ae}.log
#                done
            done
#        done
    done
done


