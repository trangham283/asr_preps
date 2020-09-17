#!/bin/bash

SPARSEVAL=/homes/ttmt001/transitory/prosodic-anomalies/SParseval
S=$SPARSEVAL/SPEECHPAR_unedit.prm
H=$SPARSEVAL/headInfo.txt

###########################################################
# test on same-word data
#DATA=/homes/ttmt001/transitory/self-attentive-parser/results
#OUT=asr_output/nbest/for_parsing
##SPLIT=test
##MODEL=3704
#
#for SPLIT in dev test
#do
#    for MODEL in 1700 1701 1702 1703 3700 3701 3702 3703
#    do
#        PRED=$DATA/bert/${SPLIT}_bert_freeze_${MODEL}_predicted.txt
#        GOLD=$DATA/swbd_${SPLIT}_gold.txt 
#        $SPARSEVAL/src/sparseval -p $S -h $H -v $GOLD $PRED > $OUT/${SPLIT}_labeled_bradep_oracle_${MODEL}_unedit.log
#        $SPARSEVAL/src/sparseval -p $S -h $H -v -u $GOLD $PRED > $OUT/${SPLIT}_unlabeled_bradep_oracle_${MODEL}_unedit.log
#    done
#done

###########################################################
# test on ASR data, no alignments
#SPLIT=dev
#MODEL=1704
#DATA=asr_output/nbest/for_parsing
#OUT=asr_output/nbest
#PRED=$DATA/${SPLIT}_asr_${MODEL}.parse
#GOLD=$DATA/${SPLIT}_asr_mrg.txt 

#$SPARSEVAL/src/sparseval -p $S -h $H -v -b $GOLD $PRED > $OUT/${SPLIT}_labeled_dep_${MODEL}.log
#$SPARSEVAL/src/sparseval -p $S -h $H -v -b -u $GOLD $PRED > $OUT/${SPLIT}_unlabeled_dep_${MODEL}.log

###########################################################
# test on ASR data, with alignments
# num_end = 44422 for dev; 44237 for test
#MODEL=3704

SPLIT=dev
DATA=/s0/ttmt001/sparseval_aligns
OUT=/s0/ttmt001/sparseval_aligns/logs_dev

for MODEL in 1700 1701 1702 1703 3700 3701 3702 3703
do
    for NUM in `seq 0 44422`
    do
        PRED=$DATA/$SPLIT/pred_${MODEL}_${NUM}.mrg
        GOLD=$DATA/$SPLIT/gold_${NUM}.mrg 
        ALIGN=$DATA/$SPLIT/align_${MODEL}_${NUM}.mrg

        $SPARSEVAL/src/sparseval -p $S -h $H -v -a $ALIGN $GOLD $PRED > $OUT/${SPLIT}_labeled_bradep_${MODEL}_unedit_${NUM}.log
        $SPARSEVAL/src/sparseval -p $S -h $H -v -u -a $ALIGN $GOLD $PRED > $OUT/${SPLIT}_unlabeled_bradep_${MODEL}_unedit_${NUM}.log
    done
done
