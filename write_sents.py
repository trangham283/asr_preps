#!/usr/bin/env python3
import sys, argparse, os
from difflib import SequenceMatcher
import pandas as pd 
import numpy as np
import pstree

def calc_prf(match, gold, test):
    precision = match / float(test)
    recall = match / float(gold)
    fscore = 2 * match / (float(test + gold))
    return precision, recall, fscore
    
def make_sent_id(file_num, speaker, turn, sent_num):
    sent_id = "{}_{}{}_{}".format(file_num, speaker, turn, sent_num)
    return sent_id

# write this sentence-by-sentece 
def write_by_sent(ref_file, pred_file, model):
    ref_parse = open(ref_file).readlines()
    ref_parse = [x.strip() for x in ref_parse]
    pred_parse = open(pred_file).readlines()
    pred_parse = [x.strip() for x in pred_parse]
    assert len(ref_parse) == len(pred_parse)

    for i in range(len(ref_parse)):
        fout_gold = open("{}/gold_{}.mrg".format(out_dir, i), 'w') 
        fout_pred = open("{}/pred_{}_{}.mrg".format(out_dir, model, i), 'w') 
        fout_align = open("{}/align_{}_{}.mrg".format(out_dir, model, i), 'w') 
        ref_sent = ref_parse[i]
        ref_tree = pstree.tree_from_text(ref_sent)
        fout_gold.write(str(ref_tree))
        ref_toks = ref_tree.word_yield(as_list=True)
        pred_sent = pred_parse[i]
        pred_tree = pstree.tree_from_text(pred_sent, allow_empty_labels=True)
        fout_pred.write(str(pred_tree))
        try:
            pred_toks = pred_tree.word_yield(as_list=True)
        except:
            print(i, ref_parse, pred_parse)

        # .get_opcodes returns ops to turn a into b 
        sseq = SequenceMatcher(None, ref_toks, pred_toks)
        for tag, i1, i2, j1, j2 in sseq.get_opcodes():
            left = range(i1, i2)
            right = range(j1, j2)
            if tag == 'equal':
                for k in range(len(left)):
                    fout_align.write("{}\t{}\t{}".format(
                        ref_toks[left[k]], pred_toks[right[k]], '000\n'))
            elif tag == 'insert':
                for k in range(len(right)):
                    fout_align.write("{}\t{}\t{}".format('', 
                        pred_toks[right[k]], '010\n'))
            elif tag == 'delete':
                for k in range(len(left)):
                    fout_align.write("{}\t{}\t{}".format(
                        ref_toks[left[k]], '', '100\n'))
            else:
                # replace:
                if len(left) == len(right):
                    # same number of substitutions
                    for k in range(len(left)):
                        fout_align.write("{}\t{}\t{}".format(
                            ref_toks[left[k]], pred_toks[right[k]], '001\n'))
                else:
                    # make some insertions and deletions
                    if len(left) < len(right): 
                        # treat as insertions
                        overlap = len(left)
                        for k in range(len(right)):
                            if k < overlap:
                                fout_align.write("{}\t{}\t{}".format(
                                    ref_toks[left[k]], 
                                    pred_toks[right[k]], '001\n'))
                            else:
                                fout_align.write("{}\t{}\t{}".format('', 
                                    pred_toks[right[k]], '010\n'))
                    else:
                        # treat as deletion
                        overlap = len(right)
                        for k in range(len(left)):
                            if k < overlap:
                                fout_align.write("{}\t{}\t{}".format(
                                    ref_toks[left[k]], 
                                    pred_toks[right[k]], '001\n'))
                            else:
                                fout_align.write("{}\t{}\t{}".format(
                                    ref_toks[left[k]], '', '100\n'))

        fout_gold.close()
        fout_pred.close()
        fout_align.close()

def write_alignments(ref_file, pred_file):
    ref_parse = open(ref_file).readlines()
    ref_parse = [x.strip() for x in ref_parse]
    pred_parse = open(pred_file).readlines()
    pred_parse = [x.strip() for x in pred_parse]
    assert len(ref_parse) == len(pred_parse)

    for i in range(len(ref_parse)):
        ref_sent = ref_parse[i]
        ref_tree = pstree.tree_from_text(ref_sent)
        ref_toks = ref_tree.word_yield(as_list=True)
        pred_sent = pred_parse[i]
        pred_tree = pstree.tree_from_text(pred_sent, allow_empty_labels=True)
        try:
            pred_toks = pred_tree.word_yield(as_list=True)
        except:
            print(i, ref_parse, pred_parse)

        # .get_opcodes returns ops to turn a into b 
        sseq = SequenceMatcher(None, ref_toks, pred_toks)
        for tag, i1, i2, j1, j2 in sseq.get_opcodes():
            left = range(i1, i2)
            right = range(j1, j2)
            if tag == 'equal':
                for k in range(len(left)):
                    print("{}\t{}\t{}".format(ref_toks[left[k]], \
                            pred_toks[right[k]], '000'))
            elif tag == 'insert':
                for k in range(len(right)):
                    print("{}\t{}\t{}".format('', \
                            pred_toks[right[k]], '010'))
            elif tag == 'delete':
                for k in range(len(left)):
                    print("{}\t{}\t{}".format(ref_toks[left[k]], \
                            '', '100'))
            else:
                # replace:
                if len(left) == len(right):
                    # same number of substitutions
                    for k in range(len(left)):
                        print("{}\t{}\t{}".format(ref_toks[left[k]], \
                            pred_toks[right[k]], '001'))
                else:
                    # make some insertions and deletions
                    if len(left) < len(right): 
                        # treat as insertions
                        overlap = len(left)
                        for k in range(len(right)):
                            if k < overlap:
                                print("{}\t{}\t{}".format(ref_toks[left[k]], \
                                    pred_toks[right[k]], '001'))
                            else:
                                print("{}\t{}\t{}".format('', \
                                    pred_toks[right[k]], '010'))
                    else:
                        # treat as deletion
                        overlap = len(right)
                        for k in range(len(left)):
                            if k < overlap:
                                print("{}\t{}\t{}".format(ref_toks[left[k]], \
                                    pred_toks[right[k]], '001'))
                            else:
                                print("{}\t{}\t{}".format(ref_toks[left[k]], \
                                    '', '100'))


data_dir = "/homes/ttmt001/transitory/asr_preps/asr_output/nbest/for_parsing/"

for split in ['dev', 'test']:
    out_dir = "/s0/ttmt001/sparseval_aligns/" + split
    for model in [1700, 1701, 1702, 1703, 3700, 3701, 3702, 3703]:
        ref_file = "{}{}_asr_mrg.txt".format(data_dir, split)
        pred_file = "{}{}_asr_{}.parse".format(data_dir, split, model)
        write_by_sent(ref_file, pred_file, model)
        

