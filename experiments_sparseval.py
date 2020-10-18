#!/usr/bin/env python3

import os
import argparse
import pandas as pd
import glob
import numpy as np
import json
import jiwer
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
import random
import pickle, json
import code
from nltk.tree import Tree
from pstree import *

random.seed(0)

tag_keys = ['PRP$', 'VBG', 'VBD', '``', 'VBN', ',', "''", 'VBP', 'WDT', 'JJ', 
'WP', 'VBZ', 'DT', 'RP', '$', 'NN', ')', '(', 'FW', 'POS', '.', 'TO', 'LS', 
'RB', ':', 'NNS', 'NNP', 'VB', 'WRB', 'CC', 'PDT', 'RBS', 'RBR', 'CD', 'PRP', 
'EX', 'IN', 'WP$', 'MD', 'NNPS', '--', 'JJS', 'JJR', 'SYM', 'UH']
columns = ['sent_num', 
'bracket_match', 'bracket_gold', 'bracket_test', 'bracket_cross', 
'overall_match', 'overall_gold', 'overall_test', 
'open_match', 'open_gold', 'open_test']
# precision = match / float(test)
# recall = match / float(gold)
# fscore = 2 * match / (float(test + gold))

def comp_fsent(row, score_type):
    col_match = score_type + '_match' 
    col_gold = score_type + '_gold'
    col_test = score_type + '_test'
    if row[col_test] + row[col_gold] == 0:
        return 0.0
    fscore = 2 * row[col_match] / (row[col_test] + row[col_gold])
    return fscore

def read_result_file(filename):
    ll = open(filename).readlines()
    lines = ll[4:-22]
    lines = [x.split() for x in lines]
    df = pd.DataFrame(lines, columns=columns, dtype=float)
    print("Num sentences:", len(df))
    return df
    
def add_parses(asr_dir, split, model, df):
    pred_file = os.path.join(asr_dir, "{}_asr_{}.parse".format(split, model))
    parses = open(pred_file).readlines()
    parses = [x.strip() for x in parses]
    df['pred_parse'] = parses
    pscore_file = os.path.join(asr_dir, "{}_asr_{}.scores".format(split, model))
    pscores = open(pscore_file).readlines()
    pscores = [float(x.strip().split()[0]) for x in pscores]
    df['pscores_raw'] = pscores 
    return df

def add_f1_scores(df):
    df['overall_f1'] = df.apply(lambda x: comp_fsent(x, 'overall'), axis=1)
    df['open_f1'] = df.apply(lambda x: comp_fsent(x, 'open'), axis=1)
    df['bracket_f1'] = df.apply(lambda x: comp_fsent(x, 'bracket'), axis=1)
    return df

def get_merge_asr_df(asr_dir, split, model, dep_type, df):
    tsv_file = "asr_output/nbest/" + split + "_asr_pa_nbest.tsv"
    tsv_df = pd.read_csv(tsv_file, sep="\t")
    tsv_df = tsv_df.rename(columns={'mrg':'gold_parse'})
    tsv_df['sent_num'] = range(1, len(tsv_df)+1)
    merge_df = pd.merge(tsv_df, df, on='sent_num')
    merge_df = merge_df.sort_values('sent_num')
    merge_df['asr_hyp'] = merge_df.sent_id.apply(lambda x: int(x.split('-')[1]))
    merge_df = add_parses(asr_dir, split, model, merge_df)
    merge_df['asr_score'] = -(merge_df['lm_cost'] + 0.1*merge_df['ac_cost'])
    asr_min = merge_df.asr_score.min()
    merge_df['asr_len'] = merge_df['asr_sent'].apply(lambda x: len(x.split()))
    merge_df['asr_norm'] = (merge_df['asr_score'] - asr_min)/merge_df['asr_len']
    merge_df['wer'] = merge_df.apply(lambda row: 
            jiwer.wer(row.orig_sent, row.asr_sent), axis=1)
    pscore_min = merge_df['pscores_raw'].min()
    merge_df['parse_score'] = (merge_df['pscores_raw'] - pscore_min) / merge_df['asr_len']
    merge_df['edit_count'] = merge_df['pred_parse'].apply(lambda x: 
            x.count('EDITED'))
    merge_df['intj_count'] = merge_df['pred_parse'].apply(lambda x: 
            x.count('INTJ'))
    merge_df['np_count'] = merge_df['pred_parse'].apply(lambda x: x.count('NP'))
    merge_df['vp_count'] = merge_df['pred_parse'].apply(lambda x: x.count('VP'))
    merge_df['pp_count'] = merge_df['pred_parse'].apply(lambda x: x.count('PP'))
    merge_df['depth_proxy'] = merge_df['pred_parse'].apply(lambda x: 
            x.count('('))
    merge_df['depth'] = merge_df['pred_parse'].apply(lambda x: 
            Tree.fromstring(x).height())
    return merge_df

def get_features(asr_dir, split, model, dep_type, df, tsv_df):
    merge_df = pd.merge(tsv_df, df, on='sent_num')
    merge_df = merge_df.sort_values('sent_num')
    merge_df['asr_hyp'] = merge_df.sent_id.apply(lambda x: int(x.split('-')[1]))
    merge_df = add_parses(asr_dir, split, model, merge_df)
    merge_df['asr_score'] = -(merge_df['lm_cost'] + 0.1*merge_df['ac_cost'])
    asr_min = merge_df.asr_score.min()
    merge_df['asr_len'] = merge_df['asr_sent'].apply(lambda x: len(x.split()))
    merge_df['asr_norm'] = (merge_df['asr_score'] - asr_min)/merge_df['asr_len']
    merge_df['wer'] = merge_df.apply(lambda row: 
            jiwer.wer(row.orig_sent, row.asr_sent), axis=1)
    pscore_min = merge_df['pscores_raw'].min()
    merge_df['parse_score'] = (merge_df['pscores_raw'] - pscore_min) / merge_df['asr_len']
    merge_df['edit_count'] = merge_df['pred_parse'].apply(lambda x: 
            x.count('EDITED'))
    merge_df['intj_count'] = merge_df['pred_parse'].apply(lambda x: 
            x.count('INTJ'))
    merge_df['np_count'] = merge_df['pred_parse'].apply(lambda x: x.count('NP'))
    merge_df['vp_count'] = merge_df['pred_parse'].apply(lambda x: x.count('VP'))
    merge_df['pp_count'] = merge_df['pred_parse'].apply(lambda x: x.count('PP'))
    merge_df['depth_proxy'] = merge_df['pred_parse'].apply(lambda x: 
            x.count('('))
    merge_df['depth'] = merge_df['pred_parse'].apply(lambda x: 
            Tree.fromstring(x).height())
    return merge_df

def make_pairs_old(df, feat_list, n=5, comp_val='overall'):
    comp_val = comp_val + '_f1'
    pair_idx = {}
    X, Y, WER_diffs = [], [], []
    for orig_id, sent_df in df.groupby('orig_id'):
        pair_idx[orig_id] = []
        if len(sent_df) < 2:
            #print(orig_id)
            continue
        min_loc = sent_df[comp_val].idxmin()
        max_loc = sent_df[comp_val].idxmax()
        # always include the biggest difference:
        y = 1
        wer_delta = sent_df.loc[max_loc].wer - sent_df.loc[min_loc].wer
        x = []
        pair_idx[orig_id].append((max_loc, min_loc))
        for feat in feat_list:
            featval = sent_df.loc[max_loc][feat] - sent_df.loc[min_loc][feat]
            x.append(featval)
        Y.append(y)
        X.append(x)
        WER_diffs.append(wer_delta)
        
        # sample and compare with min, max
        for _ in range(n):
            sample_row = sent_df.sample(1)
            idx = sample_row.index.values[0]
            pair_idx[orig_id].append((max_loc, idx))
            sample_f1 = sample_row[comp_val].values[0] 
            
            diff = sent_df.loc[max_loc][comp_val] - sample_f1
            wer_delta = sent_df.loc[max_loc].wer - sample_row.wer.values[0]
            x = []
            for feat in feat_list:
                featval = sent_df.loc[max_loc][feat]-sample_row[feat].values[0]
                x.append(featval)
            if diff > 0:
                y = 1
            else:
                y = 0
            Y.append(y)
            X.append(x)
            WER_diffs.append(wer_delta)
            
            diff = sent_df.loc[min_loc][comp_val] - sample_f1
            wer_delta = sent_df.loc[min_loc].wer - sample_row.wer.values[0]
            pair_idx[orig_id].append((min_loc, idx))
            x = []
            for feat in feat_list:
                featval = sent_df.loc[min_loc][feat]-sample_row[feat].values[0]
                x.append(featval)
            
            if diff > 0:
                y = 1
            else:
                y = 0
            Y.append(y)
            X.append(x)
            WER_diffs.append(wer_delta)
        
        # sample random pair
        for _ in range(n):
            pair_df = sent_df.sample(2)
            idx = pair_df.index.values
            pair_idx[orig_id].append((idx[0], idx[1]))
            diff = pair_df[comp_val].values[0] - pair_df[comp_val].values[1]
            wer_delta = pair_df.wer.values[0] - pair_df.wer.values[1]
            x = []
            for feat in feat_list:
                featval = pair_df[feat].values[0] -  pair_df[feat].values[1]
                x.append(featval)
            if diff > 0:
                y = 1
            else:
                y = 0
            Y.append(y)
            X.append(x)
            WER_diffs.append(wer_delta)
    return np.array(X), Y, WER_diffs, pair_idx

def make_pairs(df, feat_list, n=5, comp_val='overall'):
    comp_val = comp_val + '_f1'
    pair_idx = {}
    X, Y, WER_diffs = [], [], []
    for orig_id, sent_df in df.groupby('orig_id'):
        pair_idx[orig_id] = []
        if len(sent_df) < 8:
            #print(orig_id)
            continue
        min_loc = sent_df[comp_val].idxmin()
        max_loc = sent_df[comp_val].idxmax()
        # always include the biggest difference:
        y = 1
        wer_delta = sent_df.loc[max_loc].wer - sent_df.loc[min_loc].wer
        x = []
        pair_idx[orig_id].append((max_loc, min_loc))
        for feat in feat_list:
            featval = sent_df.loc[max_loc][feat] - sent_df.loc[min_loc][feat]
            x.append(featval)
        Y.append(y)
        X.append(x)
        WER_diffs.append(wer_delta)
        
        # sample and compare with min, max
        for _ in range(n):
            sample_row = sent_df.sample(1)
            idx = sample_row.index.values[0]
            pair_idx[orig_id].append((max_loc, idx))
            sample_f1 = sample_row[comp_val].values[0] 
            
            diff = sent_df.loc[max_loc][comp_val] - sample_f1
            wer_delta = sent_df.loc[max_loc].wer - sample_row.wer.values[0]
            x = []
            for feat in feat_list:
                featval = sent_df.loc[max_loc][feat]-sample_row[feat].values[0]
                x.append(featval)
            if diff > 0:
                y = 1
            else:
                y = 0
            Y.append(y)
            X.append(x)
            WER_diffs.append(wer_delta)
            
            diff = sent_df.loc[min_loc][comp_val] - sample_f1
            wer_delta = sent_df.loc[min_loc].wer - sample_row.wer.values[0]
            pair_idx[orig_id].append((min_loc, idx))
            x = []
            for feat in feat_list:
                featval = sent_df.loc[min_loc][feat]-sample_row[feat].values[0]
                x.append(featval)
            
            if diff > 0:
                y = 1
            else:
                y = 0
            Y.append(y)
            X.append(x)
            WER_diffs.append(wer_delta)
        
        # sample random pair
        for _ in range(n):
            pair_df = sent_df.sample(2)
            idx = pair_df.index.values
            pair_idx[orig_id].append((idx[0], idx[1]))
            diff = pair_df[comp_val].values[0] - pair_df[comp_val].values[1]
            wer_delta = pair_df.wer.values[0] - pair_df.wer.values[1]
            x = []
            for feat in feat_list:
                featval = pair_df[feat].values[0] -  pair_df[feat].values[1]
                x.append(featval)
            if diff > 0:
                y = 1
            else:
                y = 0
            Y.append(y)
            X.append(x)
            WER_diffs.append(wer_delta)
    return np.array(X), Y, WER_diffs, pair_idx

def get_res(dev_df, clf, feat_list, prefix):
    current_df = dev_df.copy()
    Xdev = dev_df[feat_list].values
    pred_dev = clf.predict(Xdev) 
    pred_scores = clf.predict_proba(Xdev)
    rank_scores = pred_scores[:,1]
    current_df.loc[:,'pred_scores'] = rank_scores
    # dev based on pred scores
    col = 'pred_scores'
    idxf1 = current_df.groupby('orig_id')[col].idxmax()
    m = current_df.loc[idxf1][prefix+'_match'].sum()
    t = current_df.loc[idxf1][prefix+'_test'].sum()
    g = current_df.loc[idxf1][prefix+'_gold'].sum()
    ff_pred = 2 * m / (t + g)
    ref = current_df.loc[idxf1].orig_sent.values
    asr = current_df.loc[idxf1].asr_sent.values
    ref = [x.split() for x in ref]
    asr = [x.split() for x in asr]
    flat_ref = [item for sublist in ref for item in sublist]
    flat_asr = [item for sublist in asr for item in sublist]
    wer_pred = jiwer.wer(flat_ref, flat_asr)
    return current_df.loc[idxf1], ff_pred, wer_pred

def pred_by_pair(dev_df, clf, feat_list):  
    save_cols = ['sent_num', 
                 'overall_match', 'overall_gold', 'overall_test', 'overall_f1',
                 'bracket_match', 'bracket_gold', 'bracket_test', 'bracket_f1',
                'pred_parse', 'pscores_raw', 'sent_id', 'asr_hyp', 'orig_id',
                'start_times_asr', 'end_times_asr', 'true_speaker', 'asr_sent',
                'gold_parse', 'orig_sent', 'asr_score', 'asr_len', 
                'parse_score', 'asr_norm', 'wer']
    list_row = []
    for orig_id, sent_df in dev_df.groupby('orig_id'):
        first_row = sent_df.iloc[0]
        for i in range(1, len(sent_df)):
            x = []
            next_row = sent_df.iloc[i]
            for feat in feat_list:
                featval = next_row[feat] - first_row[feat]
                x.append(featval)
            x = np.array(x).reshape(1, len(feat_list))
            pred = clf.predict(x)
            #print(i, pred, x)
            if pred > 0:
                first_row = next_row.copy()
                del next_row    
        save_row = {}
        for col in save_cols:
            save_row.update({col: first_row[col]})
        list_row.append(save_row)
    return pd.DataFrame(list_row)

gold_dir = '/homes/ttmt001/transitory/self-attentive-parser/results'
def add_orig_ids(split, df):
    sent_id_file = os.path.join(gold_dir, 'swbd_' + split + '_sent_ids.txt')
    sent_ids = open(sent_id_file).readlines()
    sent_ids = [x.strip() for x in sent_ids]
    df['orig_id'] = sent_ids
    return df

raw_dir = gold_dir + "/bert"
def read_raw_scores(split, model):
    sent_id_file = os.path.join(gold_dir, 'swbd_' + split + '_sent_ids.txt')
    sent_ids = open(sent_id_file).readlines()
    sent_ids = [x.strip() for x in sent_ids]
    log_file = os.path.join(raw_dir, 
            "{}_bert_freeze_{}_results.txt".format(split, model))
    ll = open(log_file).read()
    results = []
    _, res, _ = ll.split("============================================================================\n")
    res = res.split('\n')
    res = [x.strip().split() for x in res]
    res = [x for x in res if x]
    assert len(res) == len(sent_ids)
    for line in res:
        sent_num, sent_len, stat, recall, precision, match, gold, test, \
                cross, w, tag, tag_accuracy = line
        test = float(test)
        gold = float(gold)
        f = 2 * float(match) / (float(test + gold))
        results.append({'f1_{}'.format(model): f})
    df = pd.DataFrame(results)
    df['orig_id'] = sent_ids
    return df


def compute_oracles(df, dep=True):
    if dep:
        prefix = 'overall'
    else:
        prefix = 'bracket'
    #for col in ['wer', 'asr_score', prefix+'_f1', 'parse_score']:
    for col in ['asr_score', prefix+'_f1']:
        if col == 'wer':
            idxf1 = df.groupby('orig_id')[col].idxmin()
        else:
            idxf1 = df.groupby('orig_id')[col].idxmax()
        m = df.loc[idxf1][prefix+'_match'].sum()
        t = df.loc[idxf1][prefix+'_test'].sum()
        g = df.loc[idxf1][prefix+'_gold'].sum()
        ff = 2 * m / (t + g)
        ref = df.loc[idxf1].orig_sent.values
        asr = df.loc[idxf1].asr_sent.values
        ref = [x.split() for x in ref]
        asr = [x.split() for x in asr]
        flat_ref = [item for sublist in ref for item in sublist]
        flat_asr = [item for sublist in asr for item in sublist]
        wer = jiwer.wer(flat_ref, flat_asr)
        print("Oracle F1 and WER by {}:\t {} {}".format(col, ff, wer))
    return

def run_oracles_median(args):
    min_model = args.min_model
    split = args.split
    dep_type = args.dep_type
    asr_dir = args.asr_dir
    with open('rank_exp_sents_split.json', 'r') as f:
        split_data = json.load(f)
    dev_sents = split_data['dev']

    add_edit = bool(args.add_edit)
    if add_edit:
        add_edit_str = "_unedit.pickle"
    else:
        add_edit_str = ".pickle"
    
    dev_name = os.path.join(asr_dir, "median_dev_{}_{}{}".format(dep_type,
        min_model, add_edit_str))
    test_name = os.path.join(asr_dir, "median_test_{}_{}{}".format(dep_type,
        min_model, add_edit_str))
    print(dev_name)
    print(test_name)

    with open(dev_name, 'rb') as f:
        all_dev_oracle, all_dev_asr = pickle.load(f)
    oracle_dev = all_dev_oracle[all_dev_oracle.orig_id.isin(dev_sents)]
    print("DEV")
    print("0-WER set:")
    for prefix in ['overall', 'bracket']:
        m = oracle_dev[prefix+'_match'].sum()
        t = oracle_dev[prefix+'_test'].sum()
        g = oracle_dev[prefix+'_gold'].sum()
        ff = 2 * m / (t + g)
        print("\t", prefix, ff)
    dev_mask = all_dev_asr.orig_id.isin(dev_sents)
    dev_df = all_dev_asr[dev_mask]
    dev_df = add_f1_scores(dev_df)

    print("ASR set:")
    print("\tfor dependencies:")
    compute_oracles(dev_df, dep=True)
    print("\tfor brackets:")
    compute_oracles(dev_df, dep=False)


    # Test set
    print("\nTEST")
    with open(test_name, 'rb') as f:
        test_oracle, test_asr = pickle.load(f)
    print("0-WER set:")
    for prefix in ['overall', 'bracket']:
        m = test_oracle[prefix+'_match'].sum()
        t = test_oracle[prefix+'_test'].sum()
        g = test_oracle[prefix+'_gold'].sum()
        ff = 2 * m / (t + g)
        print("\t", prefix, ff)
    test_df = add_f1_scores(test_asr)

    print("ASR set:")
    print("\tfor dependencies:")
    compute_oracles(test_df, dep=True)
    print("\tfor brackets:")
    compute_oracles(test_df, dep=False)
    return 


def run_oracles(args):
    with open('rank_exp_sents_split.json', 'r') as f:
        split_data = json.load(f)
    dev_sents = split_data['dev']

    # ASR files
    if bool(args.add_edit):
        filename = os.path.join(args.asr_dir, 
            "{}_{}_bradep_{}_unedit.tsv".format('dev', 
            args.dep_type, args.model))
        oraclefile = os.path.join(args.asr_dir,
            "{}_{}_bradep_oracle_{}_unedit.log".format('dev', 
            args.dep_type, args.model))
    else:
        filename = os.path.join(args.asr_dir, 
            "{}_{}_bradep_{}.tsv".format('dev', args.dep_type, args.model))
        oraclefile = os.path.join(args.asr_dir,
            "{}_{}_bradep_oracle_{}.log".format('dev',
            args.dep_type, args.model))

    oracle_df = read_result_file(oraclefile)
    oracle_df = add_orig_ids('dev', oracle_df) 
    oracle_dev = oracle_df[oracle_df.orig_id.isin(dev_sents)]
    
    print("DEV")
    print("0-WER set:")
    for prefix in ['overall', 'bracket']:
        m = oracle_dev[prefix+'_match'].sum()
        t = oracle_dev[prefix+'_test'].sum()
        g = oracle_dev[prefix+'_gold'].sum()
        ff = 2 * m / (t + g)
        print("\t", prefix, ff)
    df = pd.read_csv(filename, sep="\t")
    df = get_merge_asr_df(args.asr_dir, 'dev', args.model, args.dep_type, df)
    df = add_f1_scores(df)
    dev_mask = df.orig_id.isin(dev_sents)
    dev_df = df[dev_mask]

    print("ASR set:")
    print("\tfor dependencies:")
    compute_oracles(dev_df, dep=True)
    print("\tfor brackets:")
    compute_oracles(dev_df, dep=False)


    # Test set
    print("\nTEST")
    if bool(args.add_edit):
        filename = os.path.join(args.asr_dir, 
            "{}_{}_bradep_{}_unedit.tsv".format('test', 
            args.dep_type, args.model))
        oraclefile = os.path.join(args.asr_dir,
            "{}_{}_bradep_oracle_{}_unedit.log".format('test', 
            args.dep_type, args.model))
    else:
        filename = os.path.join(args.asr_dir, 
            "{}_{}_bradep_{}.tsv".format('test', args.dep_type, args.model))
        oraclefile = os.path.join(args.asr_dir,
            "{}_{}_bradep_oracle_{}.log".format('test', 
            args.dep_type, args.model))
    oracle_df = read_result_file(oraclefile)
    print("0-WER set:")
    for prefix in ['overall', 'bracket']:
        m = oracle_df[prefix+'_match'].sum()
        t = oracle_df[prefix+'_test'].sum()
        g = oracle_df[prefix+'_gold'].sum()
        ff = 2 * m / (t + g)
        print("\t", prefix, ff)
    df = pd.read_csv(filename, sep="\t")
    df = get_merge_asr_df(args.asr_dir, 'test', args.model, args.dep_type, df)
    df = add_f1_scores(df)

    print("ASR set:")
    print("\tfor dependencies:")
    compute_oracles(df, dep=True)
    print("\tfor brackets:")
    compute_oracles(df, dep=False)
    return
 
def train_medians_dt(args):
    dep_type = args.dep_type
    min_model = args.min_model
    asr_dir = args.asr_dir
    n = args.nsamples
    prefix = 'overall' if args.criteria=='dependency' else 'bracket'

    with open('rank_exp_sents_split.json', 'r') as f:
        split_data = json.load(f)
    dev_sents = split_data['dev']

    add_edit = bool(args.add_edit)
    if add_edit:
        add_edit_str = "_unedit.pickle"
    else:
        add_edit_str = ".pickle"
    
    dev_name = os.path.join(asr_dir, "median_dev_{}_{}{}".format(dep_type,
        min_model, add_edit_str))
    test_name = os.path.join(asr_dir, "median_test_{}_{}{}".format(dep_type,
        min_model, add_edit_str))
    print(dev_name)
    print(test_name)

    with open(dev_name, 'rb') as f:
        all_dev_oracle, all_dev_asr = pickle.load(f)
    oracle_dev = all_dev_oracle[all_dev_oracle.orig_id.isin(dev_sents)]
    all_dev_asr = add_f1_scores(all_dev_asr)
    dev_mask = all_dev_asr.orig_id.isin(dev_sents)
    dev_df = all_dev_asr[dev_mask]
    train_df = all_dev_asr[~dev_mask]

    if args.features == 'allold':
        feats = ['parse_score', 'asr_score', 'asr_len', 
                 'edit_count', 'depth_proxy', 'intj_count',
                 'np_count', 'vp_count', 'pp_count']
    elif args.features == 'fl1':    
        # fl1
        feats = ['parse_score', 'asr_score', 'asr_len', 
                 'edit_count', 'depth_proxy', 'intj_count']
    elif args.features == 'fl3':
        feats = ['parse_score', 'asr_score', 'asr_len', 'edit_count', 
                'depth', 'depth_proxy']
    elif args.features == 'fl4':
        feats = ['parse_score', 'asr_score', 'asr_len', 'edit_count',
                'depth', 'depth_proxy', 'intj_count']
    elif args.features == 'fl5':
        feats = ['parse_score', 'asr_score', 'asr_len', 'edit_count', 'depth']
    elif args.features == 'allnew':
        feats = ['parse_score', 'asr_score', 'asr_len', 'edit_count',
                'depth', 'depth_proxy', 'intj_count',  
                'np_count', 'vp_count', 'pp_count']
    elif args.features == 'fl6':
        feats = ['parse_score', 'asr_score', 'pscores_raw', 'asr_norm',
                'asr_len', 'edit_count',
                'depth', 'depth_proxy', 'intj_count',  
                'np_count', 'vp_count', 'pp_count']
    elif args.features == 'fl2':
        feats = ['parse_score', 'asr_score', 'asr_len', 'edit_count']
    else:
        print("invalid feature set")
        exit(1)

    Xtrain, Ytrain, WER_diffs, pair_idx = make_pairs(train_df, 
            feat_list=feats, n=n, comp_val=prefix)

    clfs = []
    hypers = [5, 10, 15, 20, 25]
    for C in hypers:
        clf = DecisionTreeClassifier(random_state=1, max_depth=C)
        clf.fit(Xtrain, Ytrain)
        res_df, dev_f1, dev_wer = get_res(dev_df, clf, feats, prefix)
        pred_df = pred_by_pair(dev_df, clf, feats)
        m = pred_df[prefix+'_match'].sum()
        t = pred_df[prefix+'_test'].sum()
        g = pred_df[prefix+'_gold'].sum()
        ff_pred = 2 * m / (t + g)
        ref = pred_df.orig_sent.values
        asr = pred_df.asr_sent.values
        ref = [x.split() for x in ref]
        asr = [x.split() for x in asr]
        flat_ref = [item for sublist in ref for item in sublist]
        flat_asr = [item for sublist in asr for item in sublist]
        pair_wer = jiwer.wer(flat_ref, flat_asr)
        clfs.append([clf, C, dev_f1, dev_wer, ff_pred, pair_wer, max(dev_f1, ff_pred)])
    best_row = sorted(clfs, key=lambda x: x[-1])[-1]
    clf_best, C, f1_best, wer_best, f1_pair_best, wer_pair_best, _ = best_row
    print("\nBest model:")
    print(best_row[0])
    print(best_row[1:])
    
    save_name = "exp_medians/{}_{}_{}_{}_dep-{}_edit-{}_lim8.pkl".format(
            args.classifier, args.min_model, args.features, 
            args.criteria, args.dep_type, args.add_edit)
    with open(save_name, 'wb') as f:
        pickle.dump(clf_best, f)

    data_ref = [Xtrain, Ytrain, WER_diffs, pair_idx, dev_df]
    bak_dir = "/s0/ttmt001/asr_preps_bak"
    save_name = "{}/mediandata_{}_{}_{}_{}_dep-{}_edit-{}_lim8.pkl".format(
            bak_dir, args.min_model, args.features, args.classifier, 
            args.criteria, args.dep_type, args.add_edit)
    with open(save_name, 'wb') as f:
        pickle.dump(data_ref, f)

    with open(test_name, 'rb') as f:
        test_oracle, test_df = pickle.load(f)
    test_df = add_f1_scores(test_df)

    test_res, test_f1, test_wer = get_res(test_df, clf_best, feats, prefix)
    pair_df = pred_by_pair(test_df, clf_best, feats)
    m = pair_df[prefix+'_match'].sum()
    t = pair_df[prefix+'_test'].sum()
    g = pair_df[prefix+'_gold'].sum()
    ff_pred = 2 * m / (t + g)
    ref = pair_df.orig_sent.values
    asr = pair_df.asr_sent.values
    ref = [x.split() for x in ref]
    asr = [x.split() for x in asr]
    flat_ref = [item for sublist in ref for item in sublist]
    flat_asr = [item for sublist in asr for item in sublist]
    pair_wer = jiwer.wer(flat_ref, flat_asr)
    print("Test set results:")
    print("Pred F1 & WER:\t", test_f1, test_wer)
    print("Pred (pair) F1 & WER:\t", ff_pred, pair_wer)
    return

def train_medians(args):
    dep_type = args.dep_type
    min_model = args.min_model
    asr_dir = args.asr_dir
    n = args.nsamples
    prefix = 'overall' if args.criteria=='dependency' else 'bracket'

    with open('rank_exp_sents_split.json', 'r') as f:
        split_data = json.load(f)
    dev_sents = split_data['dev']

    add_edit = bool(args.add_edit)
    if add_edit:
        add_edit_str = "_unedit.pickle"
    else:
        add_edit_str = ".pickle"
    
    dev_name = os.path.join(asr_dir, "median_dev_{}_{}{}".format(dep_type,
        min_model, add_edit_str))
    test_name = os.path.join(asr_dir, "median_test_{}_{}{}".format(dep_type,
        min_model, add_edit_str))
    print(dev_name)
    print(test_name)

    with open(dev_name, 'rb') as f:
        all_dev_oracle, all_dev_asr = pickle.load(f)
    oracle_dev = all_dev_oracle[all_dev_oracle.orig_id.isin(dev_sents)]
    all_dev_asr = add_f1_scores(all_dev_asr)
    dev_mask = all_dev_asr.orig_id.isin(dev_sents)
    dev_df = all_dev_asr[dev_mask]
    train_df = all_dev_asr[~dev_mask]

    if args.features == 'allold':
        feats = ['parse_score', 'asr_score', 'asr_len', 
                 'edit_count', 'depth_proxy', 'intj_count',
                 'np_count', 'vp_count', 'pp_count']
    elif args.features == 'fl1':    
        # fl1
        feats = ['parse_score', 'asr_score', 'asr_len', 
                 'edit_count', 'depth_proxy', 'intj_count']
    elif args.features == 'fl3':
        feats = ['parse_score', 'asr_score', 'asr_len', 'edit_count', 
                'depth', 'depth_proxy']
    elif args.features == 'fl4':
        feats = ['parse_score', 'asr_score', 'asr_len', 'edit_count',
                'depth', 'depth_proxy', 'intj_count']
    elif args.features == 'fl5':
        feats = ['parse_score', 'asr_score', 'asr_len', 'edit_count', 'depth']
    elif args.features == 'allnew':
        feats = ['parse_score', 'asr_score', 'asr_len', 'edit_count',
                'depth', 'depth_proxy', 'intj_count',  
                'np_count', 'vp_count', 'pp_count']
    elif args.features == 'fl6':
        feats = ['parse_score', 'asr_score', 'pscores_raw', 'asr_norm',
                'asr_len', 'edit_count',
                'depth', 'depth_proxy', 'intj_count',  
                'np_count', 'vp_count', 'pp_count']
    elif args.features == 'fl2':
        feats = ['parse_score', 'asr_score', 'asr_len', 'edit_count']
    else:
        print("invalid feature set")
        exit(1)

    Xtrain, Ytrain, WER_diffs, pair_idx = make_pairs(train_df, 
            feat_list=feats, n=n, comp_val=prefix)

    clfs = []
    hypers = [0.0005, 0.0001, 0.005, 0.001, 0.05, 0.01, 0.5, 0.1, 
            1.0, 5.0, 10.0, 50.0, 100.0]
    if args.classifier == 'SVC':
        for C in hypers:
            clf = SVC(probability=True, C=C, gamma='scale', 
                    random_state=1, max_iter=400)
            clf.fit(Xtrain, Ytrain)
            res_df, dev_f1, dev_wer = get_res(dev_df, clf, feats, prefix)
            pred_df = pred_by_pair(dev_df, clf, feats)
            m = pred_df[prefix+'_match'].sum()
            t = pred_df[prefix+'_test'].sum()
            g = pred_df[prefix+'_gold'].sum()
            ff_pred = 2 * m / (t + g)
            ref = pred_df.orig_sent.values
            asr = pred_df.asr_sent.values
            ref = [x.split() for x in ref]
            asr = [x.split() for x in asr]
            flat_ref = [item for sublist in ref for item in sublist]
            flat_asr = [item for sublist in asr for item in sublist]
            pair_wer = jiwer.wer(flat_ref, flat_asr)
            clfs.append([clf, C, dev_f1, dev_wer, ff_pred, pair_wer, max(dev_f1, ff_pred)])
    else: 
        for C in hypers:
            clf = LogisticRegression(random_state=1, C=C, solver='lbfgs')
            clf.fit(Xtrain, Ytrain)
            res_df, dev_f1, dev_wer = get_res(dev_df, clf, feats, prefix)
            pred_df = pred_by_pair(dev_df, clf, feats)
            m = pred_df[prefix+'_match'].sum()
            t = pred_df[prefix+'_test'].sum()
            g = pred_df[prefix+'_gold'].sum()
            ff_pred = 2 * m / (t + g)
            ref = pred_df.orig_sent.values
            asr = pred_df.asr_sent.values
            ref = [x.split() for x in ref]
            asr = [x.split() for x in asr]
            flat_ref = [item for sublist in ref for item in sublist]
            flat_asr = [item for sublist in asr for item in sublist]
            pair_wer = jiwer.wer(flat_ref, flat_asr)
            clfs.append([clf, C, dev_f1, dev_wer, ff_pred, pair_wer, max(dev_f1, ff_pred)])
    best_row = sorted(clfs, key=lambda x: x[-1])[-1]
    clf_best, C, f1_best, wer_best, f1_pair_best, wer_pair_best, _ = best_row
    print("\nBest model:")
    print(best_row[0])
    print(best_row[1:])
    
    save_name = "exp_medians/{}_{}_{}_{}_dep-{}_edit-{}_lim8.pkl".format(
            args.classifier, args.min_model, args.features, 
            args.criteria, args.dep_type, args.add_edit)
    with open(save_name, 'wb') as f:
        pickle.dump(clf_best, f)

    data_ref = [Xtrain, Ytrain, WER_diffs, pair_idx, dev_df]
    bak_dir = "/s0/ttmt001/asr_preps_bak"
    save_name = "{}/mediandata_{}_{}_{}_{}_dep-{}_edit-{}_lim8.pkl".format(
            bak_dir, args.min_model, args.features, args.classifier, 
            args.criteria, args.dep_type, args.add_edit)
    with open(save_name, 'wb') as f:
        pickle.dump(data_ref, f)

    with open(test_name, 'rb') as f:
        test_oracle, test_df = pickle.load(f)
    test_df = add_f1_scores(test_df)

    test_res, test_f1, test_wer = get_res(test_df, clf_best, feats, prefix)
    pair_df = pred_by_pair(test_df, clf_best, feats)
    m = pair_df[prefix+'_match'].sum()
    t = pair_df[prefix+'_test'].sum()
    g = pair_df[prefix+'_gold'].sum()
    ff_pred = 2 * m / (t + g)
    ref = pair_df.orig_sent.values
    asr = pair_df.asr_sent.values
    ref = [x.split() for x in ref]
    asr = [x.split() for x in asr]
    flat_ref = [item for sublist in ref for item in sublist]
    flat_asr = [item for sublist in asr for item in sublist]
    pair_wer = jiwer.wer(flat_ref, flat_asr)
    print("Test set results:")
    print("Pred F1 & WER:\t", test_f1, test_wer)
    print("Pred (pair) F1 & WER:\t", ff_pred, pair_wer)
    return


def analyze(args):
    dep_type = args.dep_type
    min_model = args.min_model
    asr_dir = args.asr_dir
    add_edit = bool(args.add_edit)
    prefix = 'overall' if args.criteria=='dependency' else 'bracket'

    model_pkl = "exp_medians/{}_{}_{}_{}_dep-{}_edit-{}_{}_spls.pkl".format(
            args.classifier, args.min_model, args.features, 
            args.criteria, args.dep_type, args.add_edit, args.nsamples)
    print("Loading model:", model_pkl)
    if 'fl1' in model_pkl:
        feats = ['parse_score', 'asr_score', 'asr_len', 
                 'edit_count', 'depth_proxy', 'intj_count']
    elif 'fl2' in model_pkl:
        feats = ['parse_score', 'asr_score', 'asr_len', 'edit_count']
    elif 'fl3' in model_pkl:
        feats = ['parse_score', 'asr_score', 'asr_len', 'edit_count', 
                'depth', 'depth_proxy']
    elif 'fl4' in model_pkl:
        feats = ['parse_score', 'asr_score', 'asr_len', 'edit_count',
                'depth', 'depth_proxy', 'intj_count']
    elif 'fl5' in model_pkl:
        feats = ['parse_score', 'asr_score', 'asr_len', 'edit_count', 'depth']
    elif 'allnew' in model_pkl:
        feats = ['parse_score', 'asr_score', 'asr_len', 'edit_count',
                'depth', 'depth_proxy', 'intj_count',  
                'np_count', 'vp_count', 'pp_count']
    elif 'fl6' in model_pkl:
        feats = ['parse_score', 'asr_score', 'pscores_raw', 'asr_norm',
                'asr_len', 'edit_count',
                'depth', 'depth_proxy', 'intj_count',  
                'np_count', 'vp_count', 'pp_count']
    elif 'allold' in model_pkl:
        feats = ['parse_score', 'asr_score', 'asr_len', 
                 'edit_count', 'depth_proxy', 'intj_count',
                 'np_count', 'vp_count', 'pp_count'] 
    else:
        print("invalid feature set")
        exit(1)
    
    with open(model_pkl, 'rb') as f:
        clf_best = pickle.load(f)

    if add_edit:
        add_edit_str = "_unedit.pickle"
    else:
        add_edit_str = ".pickle"
    
    test_name = os.path.join(asr_dir, "median_test_{}_{}{}".format(dep_type,
        min_model, add_edit_str))
    print("Analyzing: ", test_name)
    with open(test_name, 'rb') as f:
        test_oracle, test_df = pickle.load(f)
    test_df = add_f1_scores(test_df)

    test_res, test_f1, test_wer = get_res(test_df, clf_best, feats, prefix)
    pair_df = pred_by_pair(test_df, clf_best, feats)
    m = pair_df[prefix+'_match'].sum()
    t = pair_df[prefix+'_test'].sum()
    g = pair_df[prefix+'_gold'].sum()
    ff_pred = 2 * m / (t + g)
    ref = pair_df.orig_sent.values
    asr = pair_df.asr_sent.values
    ref = [x.split() for x in ref]
    asr = [x.split() for x in asr]
    flat_ref = [item for sublist in ref for item in sublist]
    flat_asr = [item for sublist in asr for item in sublist]
    pair_wer = jiwer.wer(flat_ref, flat_asr)
    print("Test set results:")
    print("Pred F1 & WER:\t", test_f1, test_wer)
    print("Pred (pair) F1 & WER:\t", ff_pred, pair_wer)
    code.interact(local=locals())
    return

def train_single(args):
    dep_type = args.dep_type
    asr_dir = args.asr_dir
    if bool(args.add_edit):
        filename = os.path.join(args.asr_dir, 
            "{}_{}_bradep_{}_unedit.tsv".format('dev', 
                args.dep_type, args.model))
    else:
        filename = os.path.join(args.asr_dir, 
            "{}_{}_bradep_{}.tsv".format('dev', args.dep_type, args.model))
    df = pd.read_csv(filename, sep="\t")
    df = get_merge_asr_df(args.asr_dir, 'dev', args.model, args.dep_type, df)
    df = add_f1_scores(df)
    n = args.nsamples
    prefix = 'overall' if args.criteria=='dependency' else 'bracket'

    with open('rank_exp_sents_split.json', 'r') as f:
        split_data = json.load(f)

    dev_sents = split_data['dev']
    dev_mask = df.orig_id.isin(dev_sents)
    dev_df = df[dev_mask]
    train_df = df[~dev_mask]

    if args.features == 'allold':
        feats = ['parse_score', 'asr_score', 'asr_len', 
                 'edit_count', 'depth_proxy', 'intj_count',
                 'np_count', 'vp_count', 'pp_count']
    elif args.features == 'fl1':    
        # fl1
        feats = ['parse_score', 'asr_score', 'asr_len', 
                 'edit_count', 'depth_proxy', 'intj_count']
    elif args.features == 'fl3':
        feats = ['parse_score', 'asr_score', 'asr_len', 'edit_count', 
                'depth', 'depth_proxy']
    elif args.features == 'fl4':
        feats = ['parse_score', 'asr_score', 'asr_len', 'edit_count',
                'depth', 'depth_proxy', 'intj_count']
    elif args.features == 'fl5':
        feats = ['parse_score', 'asr_score', 'asr_len', 'edit_count', 'depth']
    elif args.features == 'fl6':
        feats = ['parse_score', 'asr_score', 'pscores_raw', 'asr_norm',
                'asr_len', 'edit_count',
                'depth', 'depth_proxy', 'intj_count',  
                'np_count', 'vp_count', 'pp_count']
    elif args.features == 'allnew':
        feats = ['parse_score', 'asr_score', 'asr_len', 'edit_count',
                'depth', 'depth_proxy', 'intj_count',  
                'np_count', 'vp_count', 'pp_count']
    elif args.features == 'fl2':
        feats = ['parse_score', 'asr_score', 'asr_len', 'edit_count']
    else:
        print("invalid feature set")
        exit(1)

    Xtrain, Ytrain, WER_diffs, pair_idx = make_pairs(train_df, 
            feat_list=feats, n=n, comp_val=prefix)

    clfs = []
    hypers = [0.0005, 0.0001, 0.005, 0.001, 0.05, 0.01, 0.5, 0.1, 
            1.0, 5.0, 10.0, 50.0, 100.0]
    if args.classifier == 'SVC':
        for C in hypers:
            clf = SVC(probability=True, C=C, gamma='scale', 
                    random_state=1, max_iter=300)
            clf.fit(Xtrain, Ytrain)
            res_df, dev_f1, dev_wer = get_res(dev_df, clf, feats, prefix)
            pred_df = pred_by_pair(dev_df, clf, feats)
            m = pred_df[prefix+'_match'].sum()
            t = pred_df[prefix+'_test'].sum()
            g = pred_df[prefix+'_gold'].sum()
            ff_pred = 2 * m / (t + g)
            ref = pred_df.orig_sent.values
            asr = pred_df.asr_sent.values
            ref = [x.split() for x in ref]
            asr = [x.split() for x in asr]
            flat_ref = [item for sublist in ref for item in sublist]
            flat_asr = [item for sublist in asr for item in sublist]
            pair_wer = jiwer.wer(flat_ref, flat_asr)
            clfs.append([clf, C, dev_f1, dev_wer, ff_pred, pair_wer])
    else: 
        for C in hypers:
            clf = LogisticRegression(random_state=1, C=C, solver='lbfgs')
            clf.fit(Xtrain, Ytrain)
            res_df, dev_f1, dev_wer = get_res(dev_df, clf, feats, prefix)
            pred_df = pred_by_pair(dev_df, clf, feats)
            m = pred_df[prefix+'_match'].sum()
            t = pred_df[prefix+'_test'].sum()
            g = pred_df[prefix+'_gold'].sum()
            ff_pred = 2 * m / (t + g)
            ref = pred_df.orig_sent.values
            asr = pred_df.asr_sent.values
            ref = [x.split() for x in ref]
            asr = [x.split() for x in asr]
            flat_ref = [item for sublist in ref for item in sublist]
            flat_asr = [item for sublist in asr for item in sublist]
            pair_wer = jiwer.wer(flat_ref, flat_asr)
            clfs.append([clf, C, dev_f1, dev_wer, ff_pred, pair_wer])
    best_row = sorted(clfs, key=lambda x: x[2])[-1]
    clf_best, C, f1_best, wer_best, f1_pair_best, wer_pair_best = best_row
    print("\nBest model:")
    print(best_row[0])
    print(best_row[1:])
    #save_name = "exp_out/{}_{}_{}_{}_dep-{}_edit-{}_{}_spls.pkl".format(
    #        args.classifier, args.model, args.features, 
    #        args.criteria, args.dep_type, args.add_edit, n)
    save_name = "/s0/ttmt001/asr_preps_bak/{}_{}_{}_{}_dep-{}_edit-{}_{}_spls.pkl".format(
            args.classifier, args.model, args.features, 
            args.criteria, args.dep_type, args.add_edit, n)
    with open(save_name, 'wb') as f:
        pickle.dump(clf_best, f)

    data_ref = [Xtrain, Ytrain, WER_diffs, pair_idx, dev_df]
    bak_dir = "/s0/ttmt001/asr_preps_bak"
    save_name = "{}/data_{}_{}_{}_{}_dep-{}_edit-{}_{}_spls.pkl".format(
            bak_dir, args.model, args.features, args.classifier, 
            args.criteria, args.dep_type, args.add_edit, n)
    with open(save_name, 'wb') as f:
        pickle.dump(data_ref, f)

    if bool(args.add_edit):
        filename = os.path.join(args.asr_dir, 
            "{}_{}_bradep_{}_unedit.tsv".format('test', 
                args.dep_type, args.model))
    else:
        filename = os.path.join(args.asr_dir, 
            "{}_{}_bradep_{}.tsv".format('test', args.dep_type, args.model))
    df = pd.read_csv(filename, sep="\t")
    test_df = get_merge_asr_df(args.asr_dir, 'test', args.model, args.dep_type, df)
    test_df = add_f1_scores(test_df)

    test_res, test_f1, test_wer = get_res(test_df, clf_best, feats, prefix)
    pair_df = pred_by_pair(test_df, clf_best, feats)
    m = pair_df[prefix+'_match'].sum()
    t = pair_df[prefix+'_test'].sum()
    g = pair_df[prefix+'_gold'].sum()
    ff_pred = 2 * m / (t + g)
    ref = pair_df.orig_sent.values
    asr = pair_df.asr_sent.values
    ref = [x.split() for x in ref]
    asr = [x.split() for x in asr]
    flat_ref = [item for sublist in ref for item in sublist]
    flat_asr = [item for sublist in asr for item in sublist]
    pair_wer = jiwer.wer(flat_ref, flat_asr)
    print("Test set results:")
    print("Pred F1 & WER:\t", test_f1, test_wer)
    print("Pred (pair) F1 & WER:\t", ff_pred, pair_wer)
    return

def evaluate_model(args):
    model_pkl = args.model_pkl
    if 'fl1' in model_pkl:
        feats = ['parse_score', 'asr_score', 'asr_len', 
                 'edit_count', 'depth_proxy', 'intj_count']
    elif 'fl2' in model_pkl:
        feats = ['parse_score', 'asr_score', 'asr_len', 'edit_count']
    elif 'fl3' in model_pkl:
        feats = ['parse_score', 'asr_score', 'asr_len', 'edit_count', 
                'depth', 'depth_proxy']
    elif 'fl4' in model_pkl:
        feats = ['parse_score', 'asr_score', 'asr_len', 'edit_count',
                'depth', 'depth_proxy', 'intj_count']
    elif 'fl5' in model_pkl:
        feats = ['parse_score', 'asr_score', 'asr_len', 'edit_count', 'depth']
    elif 'fl6' in model_pkl:
        feats = ['parse_score', 'asr_score', 'pscores_raw', 'asr_norm',
                'asr_len', 'edit_count',
                'depth', 'depth_proxy', 'intj_count',  
                'np_count', 'vp_count', 'pp_count']
    elif 'allnew' in model_pkl:
        feats = ['parse_score', 'asr_score', 'asr_len', 'edit_count',
                'depth', 'depth_proxy', 'intj_count',  
                'np_count', 'vp_count', 'pp_count']
    elif 'allold' in model_pkl:
        feats = ['parse_score', 'asr_score', 'asr_len', 
                 'edit_count', 'depth_proxy', 'intj_count',
                 'np_count', 'vp_count', 'pp_count'] 
    else:
        print("invalid feature set")
        exit(1)
    
    with open(model_pkl, 'rb') as f:
        clf_best = pickle.load(f)

    model_name = os.path.basename(model_pkl)
    model = model_name.split('_')[1]
    if bool(args.add_edit):
        filename = os.path.join(args.asr_dir, 
            "{}_{}_bradep_{}_unedit.tsv".format('test', 
                args.dep_type, args.model))
    else:
        filename = os.path.join(args.asr_dir, 
            "{}_{}_bradep_{}.tsv".format('test', args.dep_type, args.model))
    df = pd.read_csv(filename, sep="\t")
    test_df = get_merge_asr_df(args.asr_dir, 'test', args.model, args.dep_type, df)
    test_df = add_f1_scores(test_df)
    print("Number of all sentences: ", len(test_df))
    test_res, test_f1, test_wer = get_res(test_df, clf_best, feats)
    pair_df = pred_by_pair(test_df, clf_best, feats)
    print("Number of sentences: ", len(test_res), len(pair_df))
    m = pair_df['overall_match'].sum()
    t = pair_df['overall_test'].sum()
    g = pair_df['overall_gold'].sum()
    ff_pred = 2 * m / (t + g)
    ref = pair_df.orig_sent.values
    asr = pair_df.asr_sent.values
    ref = [x.split() for x in ref]
    asr = [x.split() for x in asr]
    flat_ref = [item for sublist in ref for item in sublist]
    flat_asr = [item for sublist in asr for item in sublist]
    pair_wer = jiwer.wer(flat_ref, flat_asr)
    print("Test set results:")
    print("Pred F1 & WER:\t", test_f1, "\t", test_wer)
    print("Pred (pair) F1 & WER:\t", ff_pred, "\t", pair_wer)
    #code.interact(local=locals())
    return

def get_medians(args):
    models = range(args.min_model, args.min_model + 5)
    min_model = args.min_model
    split = args.split
    add_edit = bool(args.add_edit)
    dep_type = args.dep_type
    asr_dir = args.asr_dir
    dfs = []
    colnames = []
    tsv_file = "asr_output/nbest/" + split + "_asr_pa_nbest.tsv"
    tsv_df = pd.read_csv(tsv_file, sep="\t")
    tsv_df = tsv_df.rename(columns={'mrg':'gold_parse'})
    tsv_df['sent_num'] = range(1, len(tsv_df)+1)
    for i, model in enumerate(models):
        colnames.append("f1_{}".format(model))
        if i == 0:
            this_df = read_raw_scores(split, model)
        else:
            temp = read_raw_scores(split, model)
            this_df = pd.merge(this_df, temp, on='orig_id')
        if add_edit:
            filename = os.path.join(asr_dir, 
                "{}_{}_bradep_{}_unedit.tsv".format(split, 
                dep_type, model))
            oraclefile = os.path.join(asr_dir,
                "{}_{}_bradep_oracle_{}_unedit.log".format(split, 
                dep_type, model))
            add_edit_str = "_unedit.pickle"
        else:
            filename = os.path.join(asr_dir, 
                "{}_{}_bradep_{}.tsv".format(split, dep_type, model))
            oraclefile = os.path.join(asr_dir,
                "{}_{}_bradep_oracle_{}.log".format(split,
                dep_type, model))
            add_edit_str = ".pickle"
        oracle_df = read_result_file(oraclefile)
        oracle_df = add_orig_ids(split, oracle_df)
        oracle_df = oracle_df.set_index('orig_id')
        df = pd.read_csv(filename, sep="\t")
        print("\t", model, len(df))
        df = get_features(asr_dir, split, model, dep_type, df, tsv_df)
        dfs.append((oracle_df, df))
    
    arr = this_df[colnames].values
    median_idx = np.argsort(arr, axis=1)[:,2]
    this_df['median_idx'] = median_idx

    oracles = []
    asrs = []
    for idx, orig_id in zip(median_idx, this_df.orig_id.values):
        oracle, asr = dfs[idx]
        oracle_row = oracle.loc[orig_id] 
        oracle_row['median_idx'] = idx
        oracles.append(oracle_row)
        asr_df = asr[asr.orig_id==orig_id]
        asrs.append(asr_df)

    odf = pd.DataFrame(oracles)
    odf = odf.reset_index().rename(columns={'index': 'orig_id'})
    adf = pd.concat(asrs)
    outname = os.path.join(asr_dir, "median_{}_{}_{}{}".format(split, dep_type,
        min_model, add_edit_str))
    print(add_edit, outname)
    with open(outname, 'wb') as f:
        pickle.dump([odf, adf], f)
    return

def main():
    """main function"""
    pa = argparse.ArgumentParser(
            description='Analyze parsing on ASR results')
    pa.add_argument('--asr_dir', help='asr output dir', 
            default='asr_output/nbest/for_parsing')
    pa.add_argument('--dep_type', help='dependency type, labeled or unlabled',
            default='unlabeled')
    pa.add_argument('--classifier', help='classifier type',
            default='LR')
    pa.add_argument('--model', help='model',
            default='3704')
    pa.add_argument('--nsamples', help='number of samples to sample from',
            default=5, type=int)
    pa.add_argument('--features', help='feature list: all or fl1 or fl2', 
            default='fl2')
    pa.add_argument('--model_pkl', help='model path', default=None)
    pa.add_argument('--train', help='train model', default=0, type=int)
    pa.add_argument('--add_edit', help='whether to count edits', type=int, 
            default=0)
    pa.add_argument('--criteria', help='tune by dependency or bracket score', 
            default='dependency')
    pa.add_argument('--get_medians', help='get_medians', default=0, type=int)
    pa.add_argument('--analyze', help='analyze', default=0, type=int)
    pa.add_argument('--min_model', help='min model in set', type=int,
            default=1700)
    pa.add_argument('--split', help='split', default='dev')
    args = pa.parse_args()

    print(args.model_pkl)
    if args.model_pkl is not None:
        evaluate_model(args)
    elif bool(args.train):
        #train_single(args)
        if args.classifier == 'DT':
            train_medians_dt(args)
        else:
            train_medians(args)
    elif bool(args.get_medians):
        get_medians(args)
    elif bool(args.analyze):
        analyze(args)
    else:
        #run_oracles(args)
        run_oracles_median(args)
    exit(0)


if __name__ == '__main__':
    main()
