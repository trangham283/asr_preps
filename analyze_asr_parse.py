
# coding: utf-8

# In[1]:


# %load analyze_sparseval
#!/usr/bin/env python3

import os
import argparse
import pandas as pd
import glob
import numpy as np
import json
import jiwer
from sklearn.linear_model import Ridge
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
import random
import pickle
from nltk.tree import Tree
from pstree import *

random.seed(0)

columns = ['sent_num', 
'bracket_match', 'bracket_gold', 'bracket_test', 'bracket_cross', 
'overall_match', 'overall_gold', 'overall_test', 
'open_match', 'open_gold', 'open_test']

def comp_fsent(row, score_type):
    col_match = score_type + '_match' 
    col_gold = score_type + '_gold'
    col_test = score_type + '_test'
    if row[col_test] + row[col_gold] == 0:
        return 0.0
    fscore = 2 * row[col_match] / (row[col_test] + row[col_gold])
    return fscore

def calc_prf(match, gold, test):
    precision = match / float(test)
    recall = match / float(gold)
    fscore = 2 * match / (float(test + gold))
    return precision, recall, fscore

def read_result_file(asr_dir, split, model, dep_type):
    filename = os.path.join(asr_dir, "{}_{}_dep_{}.log".format(split, dep_type, model))
    ll = open(filename).readlines()
    lines = ll[4:-13]
    lines = [x.split() for x in lines]
    df = pd.DataFrame(lines, columns=columns, dtype=float)
    pred_file = os.path.join(asr_dir, "{}_asr_{}.parse".format(split, model))
    parses = open(pred_file).readlines()
    parses = [x.strip() for x in parses]
    df['pred_parse'] = parses
    pscore_file = os.path.join(asr_dir, "{}_asr_{}.scores".format(split, model))
    pscores = open(pscore_file).readlines()
    pscores = [float(x.strip().split()[0]) for x in pscores]
    df['pscores_raw'] = pscores 
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

def get_merge_df(asr_dir, split, model, dep_type, df):
    sent_id_file = os.path.join(asr_dir, "{}_asr_sent_ids.txt".format(split))
    tsv_file = "asr_output/nbest/" + split + "_asr_pa_nbest.tsv"
    tsv_df = pd.read_csv(tsv_file, sep="\t")
    tsv_df = tsv_df.rename(columns={'mrg':'gold_parse'})
    sent_ids = open(sent_id_file).readlines()
    sent_ids = [x.strip() for x in sent_ids]
    #df = read_result_file(asr_dir, split, model, dep_type)
    df['sent_id'] = sent_ids
    df['asr_hyp'] = df.sent_id.apply(lambda x: int(x.split('-')[1]))
    merge_df = pd.merge(df, tsv_df, on='sent_id')
    merge_df['asr_score'] = -(merge_df['lm_cost'] + 0.1*merge_df['ac_cost'])
    asr_min = merge_df.asr_score.min()
    merge_df['asr_len'] = merge_df['asr_sent'].apply(lambda x: len(x.split()))
    merge_df['asr_norm'] = (merge_df['asr_score'] - asr_min) / merge_df['asr_len']
    merge_df['wer'] = merge_df.apply(lambda row: jiwer.wer(row.orig_sent, row.asr_sent), axis=1)
    
    pscore_min = merge_df['pscores_raw'].min()
    merge_df['parse_score'] = (merge_df['pscores_raw'] - pscore_min) / merge_df['asr_len']
    merge_df['edit_count'] = merge_df['pred_parse'].apply(lambda x: x.count('EDITED'))
    merge_df['intj_count'] = merge_df['pred_parse'].apply(lambda x: x.count('INTJ'))
    merge_df['np_count'] = merge_df['pred_parse'].apply(lambda x: x.count('NP'))
    merge_df['vp_count'] = merge_df['pred_parse'].apply(lambda x: x.count('VP'))
    merge_df['pp_count'] = merge_df['pred_parse'].apply(lambda x: x.count('PP'))
    merge_df['depth_proxy'] = merge_df['pred_parse'].apply(lambda x: x.count('('))
    merge_df['depth'] = merge_df['pred_parse'].apply(lambda x: Tree.fromstring(x).height())

    # analysis by sentence:
    #merge_df.groupby('orig_id').agg(
    #        best_asr=pd.NamedAgg(column='asr_score', aggfunc='max'))
    return merge_df


# In[2]:


def make_pairs(df, feat_list, n=5):
    pair_idx = {}
    X, Y, WER_diffs = [], [], []
    for orig_id, sent_df in df.groupby('orig_id'):
        pair_idx[orig_id] = []
        if len(sent_df) < 2:
            #print(orig_id)
            continue
        min_loc = sent_df.overall_f1.idxmin()
        max_loc = sent_df.overall_f1.idxmax()
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
            sample_f1 = sample_row.overall_f1.values[0] 
            
            diff = sent_df.loc[max_loc].overall_f1 - sample_f1
            wer_delta = sent_df.loc[max_loc].wer - sample_row.wer.values[0]
            x = []
            for feat in feat_list:
                featval = sent_df.loc[max_loc][feat] - sample_row[feat].values[0]
                x.append(featval)
            if diff > 0:
                y = 1
            else:
                y = 0
            Y.append(y)
            X.append(x)
            WER_diffs.append(wer_delta)
            
            diff = sent_df.loc[min_loc].overall_f1 - sample_f1
            wer_delta = sent_df.loc[min_loc].wer - sample_row.wer.values[0]
            pair_idx[orig_id].append((min_loc, idx))
            x = []
            for feat in feat_list:
                featval = sent_df.loc[min_loc][feat] - sample_row[feat].values[0]
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
            diff = pair_df.overall_f1.values[0] - pair_df.overall_f1.values[1]
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


# In[3]:


data_dir = '/homes/ttmt001/transitory/self-attentive-parser/results/bert'
gold_dir = '/homes/ttmt001/transitory/self-attentive-parser/results'

def read_parseval_files(model_id, split):
    log_file = os.path.join(data_dir, split + '_bert_freeze_' + str(model_id) +             '_results.txt')
    decoded_file = os.path.join(data_dir, split + '_bert_freeze_' + str(model_id) +             '_predicted.txt')
    score_file = os.path.join(data_dir, split + '_bert_freeze_' + str(model_id) +             '_predicted.txt.scores')
    sent_id_file = os.path.join(gold_dir, 'swbd_' + split + '_sent_ids.txt')
    gold_file = os.path.join(gold_dir, 'swbd_' + split + '_gold.txt')
    
    sent_ids = open(sent_id_file).readlines()
    sent_ids = [x.strip() for x in sent_ids]
    decoded = open(decoded_file).readlines()
    decoded = [x.strip() for x in decoded]
    scores = open(score_file).readlines()
    scores = [x.strip().split() for x in scores]
    label = open(gold_file).readlines()
    label = [x.strip() for x in label]
    assert len(sent_ids) == len(label) == len(decoded) == len(scores)
    ll = open(log_file).read()
    results = []
    _, res, _ = ll.split("============================================================================\n")
    res = res.split('\n')
    res = [x.strip().split() for x in res]
    res = [x for x in res if x]
    assert len(res) == len(sent_ids)
    for line in res:
        sent_num, sent_len, stat, recall, precision, match, gold, test,                 cross, w, tag, tag_accuracy = line
        sent_num = int(sent_num)
        p, r, f = calc_prf(float(match), float(gold), float(test))
        wav_id = sent_ids[sent_num-1]
        pscore, oscore, _ = scores[sent_num-1]
        pscore = float(pscore)
        oscore = float(oscore)
        results.append({'sent_id': wav_id,                 'match_' + str(model_id): int(match),                 'gold_' + str(model_id): int(gold),                 'test_' + str(model_id): int(test),                 'f1_' + str(model_id): f,                 'pscore_' + str(model_id): pscore,                 'ocore_' + str(model_id): oscore,                 'label_mrg': label[sent_num-1],                 'decoded_mrg_' + str(model_id): decoded[sent_num-1]})
    results = pd.DataFrame(results)
    return results


# In[4]:


def get_res(dev_df, clf, feat_list):
    current_df = dev_df.copy()
    Xdev = dev_df[feat_list].values
    pred_dev = clf.predict(Xdev) 
    pred_scores = clf.predict_proba(Xdev)
    rank_scores = pred_scores[:,1]
    current_df.loc[:,'pred_scores'] = rank_scores
    # dev based on pred scores
    col = 'pred_scores'
    idxf1 = current_df.groupby('orig_id')[col].idxmax()
    m = current_df.loc[idxf1]['overall_match'].sum()
    t = current_df.loc[idxf1]['overall_test'].sum()
    g = current_df.loc[idxf1]['overall_gold'].sum()
    ff_pred = 2 * m / (t + g)
    print("Pred F1", ff_pred)
    return current_df.loc[idxf1], ff_pred


# In[5]:


def pred_by_pair(dev_df, clf, feat_list):  
    save_cols = ['sent_num', 
                 'overall_match', 'overall_gold', 'overall_test', 'overall_f1',
                'pred_parse', 'pscores_raw', 'sent_id', 'asr_hyp', 'orig_id',
                'start_times_asr', 'end_times_asr', 'true_speaker', 'asr_sent',
                'gold_parse', 'orig_sent', 'asr_score', 'asr_len', 'parse_score', 
                'asr_norm', 'wer']
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


# In[12]:


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
    print("Oracle F1 and WER by sent_wer:\t", ff_wer, "\t", wer)

    col = 'asr_norm'
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
    print("F1 and WER by asr_score:\t", ff_asr, "\t", wer)

    col = 'overall_f1'
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
    print("Oracle F1 and WER by overall_f1:\t", ff_oracle, "\t", wer)

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
    print("F1 and WER by parse_score:\t", ff_parse, "\t", wer)


# In[7]:


def get_node_add(nodes):
    summaries = [(x.span[0], x.span[1], x.word_yield(as_list=True)) for x in nodes]
    if len(summaries) <= 1:
        return summaries
    summaries = sorted(summaries, key=lambda x: x[0])
    merged = [summaries[0]]
    for i, j, words in summaries[1:]: 
        if i < merged[-1][1]:
            continue
        else:
            merged.append((i, j, words))
    return merged

PREFIXES = ['overall_', 'open_']
def include_edits(row):
    count_edit_gold = row.gold_parse.count('EDITED')
    count_edit_pred = row.pred_parse.count('EDITED')
    if count_edit_gold > 0 or count_edit_pred > 0:
        goldtree = tree_from_text(row.gold_parse)
        edit_nodes_gold = [x for x in goldtree.get_nodes() if x.label == 'EDITED']
        predtree = tree_from_text(row.pred_parse)
        edit_nodes_pred = [x for x in predtree.get_nodes() if x.label == 'EDITED']
        gold_spans = get_node_add(edit_nodes_gold)
        test_spans = get_node_add(edit_nodes_pred)
        
        gold_add = sum([x[1]-x[0] for x in gold_spans])
        test_add = sum([x[1]-x[0] for x in test_spans])
        gold_words = [x[-1] for x in gold_spans]
        test_words = [x[-1] for x in test_spans]
        gold_words = [item for sublist in gold_words for item in sublist]
        test_words = [item for sublist in test_words for item in sublist]
        match_add = len(set(gold_words).intersection(test_words))

        #bracket_gold = 
        #bracket_test = 
        #bracket_match = 
        #row['bracket_gold'] += bracket_gold
        #row['bracket_test'] += bracket_test
        #row['bracket_match'] += bracket_match
        
        for prefix in PREFIXES:
            row[prefix+'gold'] += gold_add
            row[prefix+'test'] += test_add
            row[prefix+'match'] += match_add
        
    return row


# In[9]:


split = 'dev'
asr_dir = 'asr_output/nbest/for_parsing'
dep_type = 'unlabeled'
model = '1704'
add_edit = True

# reading model without bracketing:
# df = read_result_file(asr_dir, split, model, dep_type)

filename = os.path.join(asr_dir, "{}_{}_bradep_{}.tsv".format(split, dep_type, model))
df = pd.read_csv(filename, sep="\t")
df = add_parses(asr_dir, split, model, df)

merge_df = get_merge_df(asr_dir, split, model, dep_type, df)
merge_df['gold_len'] = merge_df.orig_sent.apply(lambda x: len(x.split()))

if add_edit:
    merge_df = merge_df.apply(lambda row: include_edits(row), axis=1)
    assert (merge_df.gold_len != merge_df.overall_gold).sum() == 0
merge_df = add_f1_scores(merge_df)


# In[11]:


#compute_oracles(merge_df)
merge_df.iloc[120]


# In[13]:


compute_oracles(merge_df, dep=True)


# In[14]:


compute_oracles(merge_df, dep=False)


# In[ ]:


text_df = get_merge_df(asr_dir, split, '1704', dep_type)
print("text model")
compute_oracles(text_df)

speech_df = merge_df.copy()
print("speech model")
compute_oracles(speech_df)


# In[ ]:


res_dir = "/homes/ttmt001/transitory/self-attentive-parser/results"
columns = ['sent_num', 
'bracket_match', 'bracket_gold', 'bracket_test', 'bracket_cross', 
'overall_match', 'overall_gold', 'overall_test', 
'open_match', 'open_gold', 'open_test']

def read_oracle(split, model, dep_type):
    filename = "asr_output/nbest/for_parsing/{}_{}_bradep_oracle_{}.log".format(split, dep_type, model)
    pred_file = res_dir + "/bert/{}_bert_freeze_{}_predicted.txt".format(split, model)
    score_file = res_dir + "/bert/{}_bert_freeze_{}_predicted.txt.scores".format(split, model)
    sent_id_file = res_dir + "/swbd_{}_sent_ids.txt".format(split)
    gold_file = res_dir + "/swbd_{}_gold.txt".format(split)
    ll = open(filename).readlines()
    lines = ll[4:-22]
    lines = [x.split() for x in lines]
    df = pd.DataFrame(lines, columns=columns, dtype=float)
    print(len(df))
    
    parses = open(pred_file).readlines()
    parses = [x.strip() for x in parses]
    print(len(parses))
    df['pred_parse'] = parses
    
    sent_ids = open(sent_id_file).readlines()
    sent_ids = [x.strip() for x in sent_ids]
    df['orig_id'] = sent_ids
    
    scores = open(score_file).readlines()
    scores = [x.strip().split()[0] for x in scores]
    df['pscores_raw'] = scores
    
    label = open(gold_file).readlines()
    label = [x.strip() for x in label]
    df['gold_parse'] = label
    df['gold_sent'] = df.gold_parse.apply(lambda x: tree_from_text(x).word_yield(as_list=True))
    df['gold_len'] = df.gold_sent.apply(lambda x: len(x))
    assert len(sent_ids) == len(label) == len(parses) == len(scores)
    return df

split = 'test'
dep_type = 'labeled'
model = '1704'
dev_oracle = read_oracle(split, model, dep_type)
dev = dev_oracle.apply(lambda row: include_edits(row), axis=1)
assert (dev.gold_len != dev.overall_gold).sum() == 0
dev = add_f1_scores(dev)
m = dev['overall_match'].sum()
t = dev['overall_test'].sum()
g = dev['overall_gold'].sum()
ff_pred = 2 * m / (t + g)
print("Pred F1 ", ff_pred)


# # Training

# In[ ]:


orig_set = set(merge_df.orig_id)
dev_len = len(orig_set) // 4
dev_sents = set(random.sample(orig_set, dev_len))
dev_mask = merge_df.orig_id.isin(dev_sents)
dev_df = merge_df[dev_mask]
train_df = merge_df[~dev_mask]

compute_oracles(dev_df)

n = 2
all_feats = ['parse_score', 'asr_score', 'asr_len', 
             'edit_count', 'depth_proxy', 'intj_count',
             'np_count', 'vp_count', 'pp_count']


Xall, Ytrain, WER_diffs, pair_idx = make_pairs(train_df, feat_list=all_feats, n=n)

#np.histogram(WER_diffs)

feat_list = all_feats[:]
Xtrain = Xall[:, :]

#clf = LogisticRegression(random_state=1, max_iter=200, C=0.001, penalty='elasticnet', solver='saga', l1_ratio=0.5)
clf = LogisticRegression(random_state=1, C=0.0001)
#clf = SVC(probability=True, kernel='poly', degree=3, gamma='scale', C=1, random_state=1, max_iter=300)
#clf = SVC(probability=True, C=0.001, gamma='auto', random_state=1, max_iter=300)

clf.fit(Xtrain, Ytrain)

res_df, _ = get_res(dev_df, clf, feat_list)
pred_df = pred_by_pair(dev_df, clf, feat_list)
m = pred_df['overall_match'].sum()
t = pred_df['overall_test'].sum()
g = pred_df['overall_gold'].sum()
ff_pred = 2 * m / (t + g)
print("Pred F1 (pair)", ff_pred)


# # Test set

# In[ ]:


test_df = get_merge_df(asr_dir, 'test', model, dep_type)
compute_oracles(test_df)


# In[ ]:


test_res = get_res(test_df, clf, feat_list)
pair_df = pred_by_pair(test_df, clf, feat_list)
m = pair_df['overall_match'].sum()
t = pair_df['overall_test'].sum()
g = pair_df['overall_gold'].sum()
ff_pred = 2 * m / (t + g)
print("Pred F1 (pair)", ff_pred)


# # Compare/Check original sets

# In[ ]:


text_bracket_df = read_parseval_files(1704, 'dev')
speech_bracket_df = read_parseval_files(3704, 'dev')


# In[ ]:


text_df.iloc[400][['overall_match', 'overall_gold', 'overall_test', 'sent_id']]


# In[ ]:


text_bracket_df[text_bracket_df.sent_id == 'sw4519_A_0083']


# In[ ]:


to_drop = ['sent_num', 'asr_hyp', 'orig_id',
       'start_times_asr', 'end_times_asr', 
       'true_speaker', 'asr_sent',
       'lm_cost', 'ac_cost', 'gold_parse', 
       'orig_sent', 'asr_score', 'asr_len',
       'asr_norm', 'wer']

to_rename = ['overall_match', 'overall_gold', 'overall_test',
       'open_match', 'open_gold', 'open_test', 'overall_f1', 'open_f1',
       'pred_parse', 'pscores_raw', 'parse_score', 'edit_count', 'intj_count',
       'np_count', 'vp_count', 'pp_count', 'depth_proxy', 'depth']

text_names = [x+'_1704' for x in to_rename]
text_cols = dict(zip(to_rename, text_names))
speech_names = [x+'_3704' for x in to_rename]
speech_cols = dict(zip(to_rename, speech_names))

temp = text_df.drop(columns=to_drop)
temp = temp.rename(columns=text_cols)
sp = speech_df.rename(columns=speech_cols)
df = pd.merge(temp, sp, on='sent_id')
df['delta'] = df.overall_f1_3704 - df.overall_f1_1704


# In[ ]:


display_cols = ['sent_id', 'overall_f1_1704', 'overall_f1_3704', 'wer', 'delta']

temp_df = df[df.gold_parse.str.contains("EDITED")]
temp_df.sort_values('delta')[display_cols]


# In[ ]:


df.iloc[32746]


# In[ ]:


idx = 32746
print(text_df.iloc[idx].pred_parse)
print(speech_df.iloc[idx].pred_parse)
print(text_df.iloc[idx].gold_parse)

print(speech_df.iloc[idx].asr_sent)
print(text_df.iloc[idx].asr_sent)
print(text_df.iloc[idx].orig_sent)


# # Debug

# In[ ]:


text_df = get_merge_df(asr_dir, 'dev', '1704', dep_type)

X, Y, WER_diffs = [], [], []
for k in pair_idx.keys():
    sent_df = text_df[text_df.orig_id==k]
    for i0, i1 in pair_idx[k]:
        wer_delta = sent_df.loc[i0].wer - sent_df.loc[i1].wer
        diff = sent_df.loc[i0].overall_f1 - sent_df.loc[i1].overall_f1
        x = []
        for feat in feat_list:
            featval = sent_df.loc[i0][feat] -  sent_df.loc[i1][feat]
            x.append(featval)
        if diff > 0:
            y = 1
        else:
            y = 0
        WER_diffs.append(wer_delta)
        X.append(x)
        Y.append(y)
    


# In[ ]:


sum(Ytrain)



# In[ ]:


xx, yy, wers, pidx = make_pairs(merge_df.head(100), feat_list=all_feats, n=5)


# In[ ]:


df = dev_df

col = 'wer'
idxf1 = df.groupby('orig_id')[col].idxmin()
m = df.loc[idxf1]['overall_match'].sum()
t = df.loc[idxf1]['overall_test'].sum()
g = df.loc[idxf1]['overall_gold'].sum()
ff_asr = 2 * m / (t + g)
print("ASR F1", ff_asr)

ref = df.loc[idxf1].orig_sent.values
asr = df.loc[idxf1].asr_sent.values

ref = [x.split() for x in ref]
asr = [x.split() for x in asr]

flat_ref = [item for sublist in ref for item in sublist]
flat_asr = [item for sublist in asr for item in sublist]
wer = jiwer.wer(flat_ref, flat_asr)
print("WER", wer)


# In[ ]:


pidx


# In[ ]:


goldtree = tree_from_text(merge_df.iloc[idx].gold_parse)
edit_nodes_gold = [x for x in goldtree.get_nodes() if x.label == 'EDITED']
predtree = tree_from_text(merge_df.iloc[idx].pred_parse)
edit_nodes_pred = [x for x in predtree.get_nodes() if x.label == 'EDITED']


gold_spans = get_node_add(edit_nodes_gold)
test_spans = get_node_add(edit_nodes_pred)

gold_add = sum([x[1]-x[0] for x in gold_spans])
test_add = sum([x[1]-x[0] for x in test_spans])

gold_words = [x[-1] for x in gold_spans]
test_words = [x[-1] for x in test_spans]

gold_words = [item for sublist in gold_words for item in sublist]
test_words = [item for sublist in test_words for item in sublist]
match_add = len(set(gold_words).intersection(test_words))

test_spans


# In[ ]:


df.columns

