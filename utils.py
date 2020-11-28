#!/usr/bin/env python3

import os
import argparse
import pandas as pd
import glob
import numpy as np
import json
import jiwer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
import random
import pickle, json
import code
from nltk.tree import Tree
from pstree import *
from scipy import stats

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

def pkl2feats(model_pkl):
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
    return feats

def compute_oracles(df, dep=True):
    if dep:
        prefix = 'overall'
    else:
        prefix = 'bracket'
    col = 'wer'
    idxf1 = df.groupby('orig_id')[col].idxmin()
    m = df.loc[idxf1][prefix+'_match'].sum()
    t = df.loc[idxf1][prefix+'_test'].sum()
    g = df.loc[idxf1][prefix+'_gold'].sum()
    ff_wer = 2 * m / (t + g)
    ref = df.loc[idxf1].orig_sent.values
    asr = df.loc[idxf1].asr_sent.values
    ref = [x.split() for x in ref]
    asr = [x.split() for x in asr]
    flat_ref = [item for sublist in ref for item in sublist]
    flat_asr = [item for sublist in asr for item in sublist]
    wer = jiwer.wer(flat_ref, flat_asr)
    r = stats.spearmanr(df.loc[idxf1][prefix+'_f1'], df.loc[idxf1].wer)
    print("Oracle F1, WER, rho by sent_wer:\t", ff_wer, "\t", wer, "\t", r,"\n")

    col = 'asr_score'
    idxf1 = df.groupby('orig_id')[col].idxmax()
    m = df.loc[idxf1][prefix+'_match'].sum()
    t = df.loc[idxf1][prefix+'_test'].sum()
    g = df.loc[idxf1][prefix+'_gold'].sum()
    ff_asr = 2 * m / (t + g)
    ref = df.loc[idxf1].orig_sent.values
    asr = df.loc[idxf1].asr_sent.values
    ref = [x.split() for x in ref]
    asr = [x.split() for x in asr]
    flat_ref = [item for sublist in ref for item in sublist]
    flat_asr = [item for sublist in asr for item in sublist]
    wer = jiwer.wer(flat_ref, flat_asr)
    r = stats.spearmanr(df.loc[idxf1][prefix+'_f1'], df.loc[idxf1].wer)
    print("Oracle F1, WER, rho by asr_score:\t", ff_asr, "\t", wer, "\t", r,"\n")

    col = prefix+'_f1'
    idxf1 = df.groupby('orig_id')[col].idxmax()
    m = df.loc[idxf1][prefix+'_match'].sum()
    t = df.loc[idxf1][prefix+'_test'].sum()
    g = df.loc[idxf1][prefix+'_gold'].sum()
    ff_oracle = 2 * m / (t + g)
    ref = df.loc[idxf1].orig_sent.values
    asr = df.loc[idxf1].asr_sent.values
    ref = [x.split() for x in ref]
    asr = [x.split() for x in asr]
    flat_ref = [item for sublist in ref for item in sublist]
    flat_asr = [item for sublist in asr for item in sublist]
    wer = jiwer.wer(flat_ref, flat_asr)
    r = stats.spearmanr(df.loc[idxf1][prefix+'_f1'], df.loc[idxf1].wer)
    print("Oracle F1, WER, rho by overall_f1:\t",ff_oracle,"\t",wer,"\t",r,"\n")

    col = 'parse_score'
    idxf1 = df.groupby('orig_id')[col].idxmax()
    m = df.loc[idxf1][prefix+'_match'].sum()
    t = df.loc[idxf1][prefix+'_test'].sum()
    g = df.loc[idxf1][prefix+'_gold'].sum()
    ff_parse = 2 * m / (t + g)
    ref = df.loc[idxf1].orig_sent.values
    asr = df.loc[idxf1].asr_sent.values
    ref = [x.split() for x in ref]
    asr = [x.split() for x in asr]
    flat_ref = [item for sublist in ref for item in sublist]
    flat_asr = [item for sublist in asr for item in sublist]
    wer = jiwer.wer(flat_ref, flat_asr)
    r = stats.spearmanr(df.loc[idxf1][prefix+'_f1'], df.loc[idxf1].wer)
    print("F1, WER, rho by parse_score:\t", ff_parse, "\t", wer, "\t", r)

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

def pred_by_pair(dev_df, clf, feat_list, prefix):  
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
    pair_df = pd.DataFrame(list_row)
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
    return pair_df, ff_pred, pair_wer


