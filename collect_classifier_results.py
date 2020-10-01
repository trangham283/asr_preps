#!/usr/bin/env python3
import argparse, os, glob
import pandas as pd 
import numpy as np

def collect_results(data_dir, feats, dep_type, classifier):
    table_dep = np.zeros((4,4))
    table_bracket = np.zeros((4, 4))
    criteria = 'dependency'
    for idx, model in enumerate([1704, 3704]):
        filename = "{}{}_{}_{}_{}_{}.log".format(data_dir, classifier, 
                criteria, model, feats, dep_type)
        ll = open(filename).readlines()
        dev = [x for x in ll if x.startswith('[')][0].strip()
        dev = dev[1:-1]
        _, pred1, wer1, pred2, wer2 = dev.split(',')
        table_dep[2*idx, 0] = float(pred1)
        table_dep[2*idx, 1] = float(wer1)
        table_dep[2*idx+1, 0] = float(pred2)
        table_dep[2*idx+1, 1] = float(wer2)
        test = [x for x in ll if x.startswith("Pred ")]
        test1, test2 = test
        t1, w1 = test1.strip().split('\t')[1].split()
        t2, w2 = test2.strip().split('\t')[1].split()
        table_dep[2*idx, 2] = float(t1)
        table_dep[2*idx, 3] = float(w1)
        table_dep[2*idx+1, 2] = float(t2)
        table_dep[2*idx+1, 3] = float(w2)
    criteria = 'bracket'
    for idx, model in enumerate([1704, 3704]):
        filename = "{}{}_{}_{}_{}_{}.log".format(data_dir, classifier, 
                criteria, model, feats, dep_type)
        ll = open(filename).readlines()
        dev = [x for x in ll if x.startswith('[')][0].strip()
        dev = dev[1:-1]
        _, pred1, wer1, pred2, wer2 = dev.split(',')
        table_bracket[2*idx, 0] = float(pred1)
        table_bracket[2*idx, 1] = float(wer1)
        table_bracket[2*idx+1, 0] = float(pred2)
        table_bracket[2*idx+1, 1] = float(wer2)
        test = [x for x in ll if x.startswith("Pred ")]
        test1, test2 = test
        t1, w1 = test1.strip().split('\t')[1].split()
        t2, w2 = test2.strip().split('\t')[1].split()
        table_bracket[2*idx, 2] = float(t1)
        table_bracket[2*idx, 3] = float(w1)
        table_bracket[2*idx+1, 2] = float(t2)
        table_bracket[2*idx+1, 3] = float(w2)
    return table_dep, table_bracket

def collect_results_median(data_dir, feats, dep_type, classifier, add_edit):
    table_dep = np.zeros((4,4))
    table_bracket = np.zeros((4, 4))
    criteria = 'dependency'
    for idx, model in enumerate([1700, 3700]):
        filename = "{}{}_{}_{}_{}_{}_edit{}.log".format(data_dir, classifier, 
                criteria, model, feats, dep_type, add_edit)
        ll = open(filename).readlines()
        dev = [x for x in ll if x.startswith('[')][0].strip()
        dev = dev[1:-1]
        _, pred1, wer1, pred2, wer2 = dev.split(',')
        table_dep[2*idx, 0] = float(pred1)
        table_dep[2*idx, 1] = float(wer1)
        table_dep[2*idx+1, 0] = float(pred2)
        table_dep[2*idx+1, 1] = float(wer2)
        test = [x for x in ll if x.startswith("Pred ")]
        test1, test2 = test
        t1, w1 = test1.strip().split('\t')[1].split()
        t2, w2 = test2.strip().split('\t')[1].split()
        table_dep[2*idx, 2] = float(t1)
        table_dep[2*idx, 3] = float(w1)
        table_dep[2*idx+1, 2] = float(t2)
        table_dep[2*idx+1, 3] = float(w2)
    criteria = 'bracket'
    for idx, model in enumerate([1700, 3700]):
        filename = "{}{}_{}_{}_{}_{}_edit{}.log".format(data_dir, classifier, 
                criteria, model, feats, dep_type, add_edit)
        ll = open(filename).readlines()
        dev = [x for x in ll if x.startswith('[')][0].strip()
        dev = dev[1:-1]
        _, pred1, wer1, pred2, wer2 = dev.split(',')
        table_bracket[2*idx, 0] = float(pred1)
        table_bracket[2*idx, 1] = float(wer1)
        table_bracket[2*idx+1, 0] = float(pred2)
        table_bracket[2*idx+1, 1] = float(wer2)
        test = [x for x in ll if x.startswith("Pred ")]
        test1, test2 = test
        t1, w1 = test1.strip().split('\t')[1].split()
        t2, w2 = test2.strip().split('\t')[1].split()
        table_bracket[2*idx, 2] = float(t1)
        table_bracket[2*idx, 3] = float(w1)
        table_bracket[2*idx+1, 2] = float(t2)
        table_bracket[2*idx+1, 3] = float(w2)
    return table_dep, table_bracket

#data_dir = "logs/"
data_dir = "logs_medians/"
add_edit = 1 
feats = 'fl1'
#classifier = 'LR'
#dep_type = 'unlabeled'

for classifier in ['LR', 'SVC']:
    for dep_type in ['unlabeled', 'labeled']:
        #td, tb = collect_results(data_dir, feats, dep_type, classifier)
        td, tb = collect_results_median(data_dir, feats, dep_type, classifier, add_edit)
        for i in range(4): 
            print(" ".join([str(x) for x in td[i, :]]))

        print()
        for i in range(4): 
            print(" ".join([str(x) for x in tb[i, :]]))

        print()

    print("\n")



