#/bin/env python3

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from scipy.stats import ttest_rel, ttest_ind

def calc_prf(match, gold, test):
    precision = match / float(test)
    recall = match / float(gold)
    fscore = 2 * match / (float(test + gold))
    return precision, recall, fscore

def bootstrap_macro(df_diff, B, col):
    n = len(df_diff)
    score = 0
    bootstrap_dist = []
    d_orig = df_diff[col].mean()
    set1 = df_diff[col].values
    for b in range(B):
        differences = []
        idx = np.random.choice(n, n)
        for i in idx:
            diff1 = set1[i]
            differences.append(diff1)
        d_new = np.mean(differences)
        bootstrap_dist.append(d_new)
        if d_new > 0:
            score += 1
    b_mean = np.mean(bootstrap_dist)
    p_count = len([x for x in bootstrap_dist if x - b_mean >= d_orig])
    p_val = (p_count+1)/float(B+1)
    print("\t# of times Sys 2 is better = {}, out of {} bootstrap samples".format(score, B))
    print("\tp-val = {}; bootstrap mean = {}".format(p_val, b_mean))
    return p_val, 1-score/float(B)

def bootstrap_micro(df, m1_name, m2_name, B, prefix):
    score = 0
    match1 = df[prefix+'match_' + m1_name].sum()
    test1 = df[prefix+'test_' + m1_name].sum()
    gold1 = df[prefix+'gold_' + m1_name].sum()
    f1 = 2 * match1 / (float(test1 + gold1))
    match2 = df[prefix+'match_' + m2_name].sum()
    test2 = df[prefix+'test_' + m2_name].sum()
    gold2 = df[prefix+'gold_' + m2_name].sum()
    f2 = 2 * match2 / (float(test2 + gold2))
    d_orig = f2 - f1
    print("\t# sentences: ", len(df))
    print("\td_orig = ", d_orig, f2, f1)
    n = len(df)
    set1 = list(zip(df[prefix+'match_' + m1_name], df[prefix+'gold_' + m1_name],             df[prefix+'test_' + m1_name]))
    set2 = list(zip(df[prefix+'match_' + m2_name], df[prefix+'gold_' + m2_name],             df[prefix+'test_' + m2_name]))
    bootstrap_dist = []
    for b in range(B):
        idx = np.random.choice(n, n)
        mm1, gm1, tm1 = 0, 0, 0
        mm2, gm2, tm2 = 0, 0, 0
        for i in idx:
            mm1 += set1[i][0]
            gm1 += set1[i][1]
            tm1 += set1[i][2]
            mm2 += set2[i][0]
            gm2 += set2[i][1]
            tm2 += set2[i][2]
        this_f1 = 2 * mm1 / (gm1 + tm1)
        this_f2 = 2 * mm2 / (gm2 + tm2)
        d_new = this_f2 - this_f1
        bootstrap_dist.append(d_new)
        if d_new > 0:
            score += 1
    b_mean = np.mean(bootstrap_dist)
    p_count = len([x for x in bootstrap_dist if x - b_mean >= d_orig])
    p_val = (p_count+1)/float(B+1)
    print("\t# of times Sys 2 is better = {}, out of {} bootstrap samples".format(score, B))
    print("\tp-val = {}; bootstrap mean = {}".format(p_val, b_mean))
    return p_val, 1-score/float(B)

# MAIN
# NOTE: These are all from LR, 'bracket', 'labeled', 'add_edit1', fl6
prefix = 'bracket'
text_df = pd.read_csv('text_df.tsv', sep="\t")
speech_df = pd.read_csv('speech_df.tsv', sep="\t")

pair_text_df = pd.read_csv('text_pair_df.tsv', sep="\t")
pair_speech_df = pd.read_csv('speech_pair_df.tsv', sep="\t")   
    
idxf1 = speech_df.groupby('orig_id')['asr_score'].idxmax()
speech_asr_df = speech_df.loc[idxf1]

idxf1 = text_df.groupby('orig_id')['asr_score'].idxmax()
text_asr_df = text_df.loc[idxf1]

ap = speech_asr_df[['orig_id', prefix+'_f1',
    prefix+'_match', prefix+'_gold', prefix+'_test','wer']]
at = text_asr_df[['orig_id', prefix+'_f1', 
    prefix+'_match', prefix+'_gold', prefix+'_test', 'wer']]
bp = pair_speech_df[['orig_id', prefix+'_f1', 
    prefix+'_match', prefix+'_gold', prefix+'_test', 'wer']]
bt = pair_text_df[['orig_id', prefix+'_f1', 
    prefix+'_match', prefix+'_gold', prefix+'_test', 'wer']]

ap = ap.rename(columns={'wer': 'wer_1best_speech', 
    prefix+'_f1': prefix+'_f1_1best_speech',
    prefix+'_match': prefix+'_match_1best_speech',
    prefix+'_test': prefix+'_test_1best_speech',
    prefix+'_gold': prefix+'_gold_1best_speech',
    })
at = at.rename(columns={'wer': 'wer_1best_text', 
    prefix+'_f1': prefix+'_f1_1best_text',
    prefix+'_match': prefix+'_match_1best_text',
    prefix+'_test': prefix+'_test_1best_text',
    prefix+'_gold': prefix+'_gold_1best_text',
    })
bp = bp.rename(columns={'wer': 'wer_classify_speech', 
    prefix+'_f1': prefix+'_f1_classify_speech',
    prefix+'_match': prefix+'_match_classify_speech',
    prefix+'_test': prefix+'_test_classify_speech',
    prefix+'_gold': prefix+'_gold_classify_speech',
    })
bt = bt.rename(columns={'wer': 'wer_classify_text', 
    prefix+'_match': prefix+'_match_classify_text',
    prefix+'_test': prefix+'_test_classify_text',
    prefix+'_gold': prefix+'_gold_classify_text',
    prefix+'_f1': prefix+'_f1_classify_text'})

m = pd.merge(ap, at, on='orig_id')
m = pd.merge(m, bp, on='orig_id')
m = pd.merge(m, bt, on='orig_id')

# NOTE: baseline is 1best_text
m['diff_wer_1best_sp'] = m['wer_1best_speech'] - m['wer_1best_text']
m['diff_f1_1best_sp'] = m[prefix+'_f1_1best_speech'] - m[prefix+'_f1_1best_text']
m['diff_wer_classify_sp'] = m['wer_classify_speech'] - m['wer_classify_text']
m['diff_f1_classify_sp'] = m[prefix+'_f1_classify_speech'] - m[prefix+'_f1_classify_text']

m['diff_speech_wer'] = m['wer_classify_speech'] - m['wer_1best_text']
m['diff_speech_f1'] = m[prefix+'_f1_classify_speech'] - m[prefix+'_f1_1best_text']
m['diff_text_wer'] = m['wer_classify_text'] - m['wer_1best_text']
m['diff_text_f1'] = m[prefix+'_f1_classify_text'] - m[prefix+'_f1_1best_text']



# Bootstrap tests
B = 100000
print("Micro boostrap, speech vs. 1best:")
pmicro, _ = bootstrap_micro(m, '1best_text', 'classify_speech', B, prefix+'_')
print("\t", pmicro)

print("Macro boostrap, speech vs. 1best:")
pmacro, _ = bootstrap_macro(m, B, 'diff_speech_f1')
print("\t", pmacro)

print("Micro boostrap, text vs. 1best:")
pmicro, _ = bootstrap_micro(m, '1best_text', 'classify_text', B, prefix+'_')
print("\t", pmicro)

print("Macro boostrap, text vs. 1best:")
pmacro, _ = bootstrap_macro(m, B, 'diff_text_f1')
print("\t", pmacro)

print("Micro boostrap, speech vs. text:")
pmicro, _ = bootstrap_micro(m, 'classify_text', 'classify_speech', B, prefix+'_')
print("\t", pmicro)

print("Macro boostrap, speech vs. text:")
pmacro, _ = bootstrap_macro(m, B, 'diff_f1_classify_sp')
print("\t", pmacro)


