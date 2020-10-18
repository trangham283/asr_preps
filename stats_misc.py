#!/usr/bin/env python3

import random
from utils import *

random.seed(0)

def print_correlation(text_df, speech_df):
    # Correlation with wer (all, including 10-hyp)
    print("Correlation on the whole set (inclde all 10 hypotheses)")
    r_sp = stats.spearmanr(speech_df.bracket_f1, speech_df.wer)
    r_tt = stats.spearmanr(text_df.bracket_f1, text_df.wer)
    print("Text, Speech:", r_tt, r_sp)
    compute_oracles(text_df, dep=False)
    compute_oracles(speech_df, dep=False)

def bin2id(asr_len):
    if asr_len == 1:
        return 1
    elif asr_len <= 5:
        return 2
    elif asr_len <= 10:
        return 3
    elif asr_len <= 15:
        return 4
    elif asr_len <= 20:
        return 5
    elif asr_len <= 40:
        return 6
    elif asr_len <= 60:
        return 7
    else:
        return 8

def get_f1(df, prefix):
    m = df[prefix+'_match'].sum()
    t = df[prefix+'_test'].sum()
    g = df[prefix+'_gold'].sum()
    ff = 2 * m / (t + g)
    return ff

dep_type = 'labeled'
asr_dir = "asr_output/nbest/for_parsing"
add_edit = 1
criteria = 'bracket'
prefix = 'overall' if criteria=='dependency' else 'bracket'
nsamples = 5
classifier = 'LR'
if bool(add_edit):
    add_edit_str = "_unedit.pickle"
else:
    add_edit_str = ".pickle"

features_text = 'fl6'
model_text = "exp_medians/{}_{}_{}_{}_dep-{}_edit-{}_lim8.pkl".format(
        classifier, 1700, features_text, 
        criteria, dep_type, add_edit)
feats_text = pkl2feats(model_text)

features_speech = 'fl6'
model_speech = "exp_medians/{}_{}_{}_{}_dep-{}_edit-{}_lim8.pkl".format(
        classifier, 3700, features_speech, 
        criteria, dep_type, add_edit)
feats_speech = pkl2feats(model_speech)

text_name = os.path.join(asr_dir, "median_test_{}_{}{}".format(dep_type,
    1700, add_edit_str))
with open(text_name, 'rb') as f:
    text_oracle, text_df = pickle.load(f)

text_df = add_f1_scores(text_df)

speech_name = os.path.join(asr_dir, "median_test_{}_{}{}".format(dep_type,
    3700, add_edit_str))
with open(speech_name, 'rb') as f:
    speech_oracle, speech_df = pickle.load(f)

speech_df = add_f1_scores(speech_df)

with open(model_text, 'rb') as f:
    clf_text = pickle.load(f)

with open(model_speech, 'rb') as f:
    clf_speech = pickle.load(f)

lens = {}
id2len = {}
for orig_id, sent_df in speech_df.groupby('orig_id'):
    num_hyp = len(sent_df)
    if num_hyp not in lens:
        lens[num_hyp] = []
    lens[num_hyp].append(orig_id)
    id2len[orig_id] = num_hyp

speech_df['num_hyp'] = speech_df.orig_id.apply(lambda x: id2len[x])
speech_df['bin_num'] = speech_df.asr_len.apply(bin2id)
text_df['num_hyp'] = text_df.orig_id.apply(lambda x: id2len[x])
text_df['bin_num'] = text_df.asr_len.apply(bin2id)

test_df = text_df
clf = clf_text
point_text_df, _, _ = get_res(test_df, clf, feats_text, prefix) 
pair_text_df, _, _ = pred_by_pair(test_df, clf, feats_text, prefix)
r_point = stats.spearmanr(point_text_df[prefix+'_f1'], point_text_df.wer)
r_pair = stats.spearmanr(pair_text_df[prefix+'_f1'], pair_text_df.wer)
print(r_point, r_pair)

test_df = speech_df
clf = clf_speech
point_speech_df, _, _ = get_res(test_df, clf, feats_speech, prefix) 
pair_speech_df, _, _ = pred_by_pair(test_df, clf, feats_speech, prefix)
r_point = stats.spearmanr(point_speech_df[prefix+'_f1'], point_speech_df.wer)
r_pair = stats.spearmanr(pair_speech_df[prefix+'_f1'], pair_speech_df.wer)
print(r_point, r_pair)


pair_speech_df['num_hyp'] = pair_speech_df.orig_id.apply(lambda x: id2len[x])
pair_speech_df['bin_num'] = pair_speech_df.asr_len.apply(bin2id)
pair_text_df['num_hyp'] = pair_text_df.orig_id.apply(lambda x: id2len[x])
pair_text_df['bin_num'] = pair_text_df.asr_len.apply(bin2id)

###################################
# write for quick access:
outname = "/s0/ttmt001/speech_df.tsv"
speech_df.to_csv(outname, sep="\t", index=False)
outname = "/s0/ttmt001/text_df.tsv"
text_df.to_csv(outname, sep="\t", index=False)
outname = "/s0/ttmt001/speech_pair_df.tsv"
pair_speech_df.to_csv(outname, sep="\t", index=False)
outname = "/s0/ttmt001/text_pair_df.tsv"
pair_text_df.to_csv(outname, sep="\t", index=False)

###################################
idxf1 = speech_df.groupby('orig_id')['asr_score'].idxmax()
speech_asr_df = speech_df.loc[idxf1]

idxf1 = text_df.groupby('orig_id')['asr_score'].idxmax()
text_asr_df = text_df.loc[idxf1]

###################################
# length stats
col1 = 'num_hyp'
col2 = 'asr_len'
last_num = 11
for i in range(1, last_num):
    sent_df = speech_df[speech_df[col1] == i]
    print(i, len(set(sent_df.orig_id)), 
            sent_df[col2].mean(), sent_df[col2].median(),
            sent_df[col2].min(), sent_df[col2].max())

col2 = 'num_hyp'
col1 = 'bin_num'
last_num = 9
for i in range(1, last_num):
    sent_df = speech_df[speech_df[col1] == i]
    print(i, len(set(sent_df.orig_id)), 
            sent_df[col2].mean(), sent_df[col2].median(),
            sent_df[col2].min(), sent_df[col2].max())

###################################
last_num = 9
col = 'bin_num'

#this_df = speech_df
#this_df = speech_asr_df
this_df = pair_speech_df

# F1 
for i in range(1, last_num):
    sent_df = this_df[this_df[col] == i]
    micro_f1 = get_f1(sent_df, prefix)
    macro_f1 = sent_df[prefix+'_f1'].mean()
    print(i, len(sent_df), macro_f1, micro_f1)

# WER
for i in range(1, last_num):
    sent_df = this_df[this_df[col] == i]
    ref = sent_df.orig_sent.to_list()
    asr = sent_df.asr_sent.to_list()
    micro_wer = jiwer.wer(ref, asr)
    macro_wer = sent_df.wer.mean()
    print(i, len(sent_df), macro_wer, micro_wer)

# By coarse bins:

# F1
split = 2
df1 = this_df[this_df[col] <= split]
df2 = this_df[this_df[col] > split]
for sent_df in [df1, df2]:
    micro_f1 = get_f1(sent_df, prefix)
    macro_f1 = sent_df[prefix+'_f1'].mean()
    print(len(sent_df), macro_f1, micro_f1)
    ref = sent_df.orig_sent.to_list()
    asr = sent_df.asr_sent.to_list()
    micro_wer = jiwer.wer(ref, asr)
    macro_wer = sent_df.wer.mean()
    print(len(sent_df), macro_wer, micro_wer)
    print()


###################################

ap = speech_asr_df[['orig_id', prefix+'_f1', 'wer']]
at = text_asr_df[['orig_id', prefix+'_f1', 'wer']]
bp = pair_speech_df[['orig_id', prefix+'_f1', 'wer']]
bt = pair_text_df[['orig_id', prefix+'_f1', 'wer']]

ap = ap.rename(columns={'wer': 'wer_1best_speech', 
    prefix+'_f1': prefix+'_f1_1best_speech'})
at = at.rename(columns={'wer': 'wer_1best_text', 
    prefix+'_f1': prefix+'_f1_1best_text'})
bp = bp.rename(columns={'wer': 'wer_classify_speech', 
    prefix+'_f1': prefix+'_f1_classify_speech'})
bt = bt.rename(columns={'wer': 'wer_classify_text', 
    prefix+'_f1': prefix+'_f1_classify_text'})

m = pd.merge(ap, at, on='orig_id')
m = pd.merge(m, bp, on='orig_id')
m = pd.merge(m, bt, on='orig_id')

m['diff_wer_1best_sp'] = m['wer_1best_speech'] - m['wer_1best_text']
m['diff_f1_1best_sp'] = m[prefix+'_f1_1best_speech'] - m[prefix+'_f1_1best_text']
m['diff_wer_classify_sp'] = m['wer_classify_speech'] - m['wer_classify_text']
m['diff_f1_classify_sp'] = m[prefix+'_f1_classify_speech'] - m[prefix+'_f1_classify_text']

m['diff_speech_wer'] = m['wer_classify_speech'] - m['wer_1best_speech']
m['diff_speech_f1'] = m[prefix+'_f1_classify_speech'] - m[prefix+'_f1_1best_speech']
m['diff_text_wer'] = m['wer_classify_text'] - m['wer_1best_text']
m['diff_text_f1'] = m[prefix+'_f1_classify_text'] - m[prefix+'_f1_1best_text']



hif1_lowe_sp = m[(m['diff_speech_f1'] > 0) & (m['diff_speech_wer'] > 0)].orig_id
lof1_hiwe_sp = m[(m['diff_speech_f1'] < 0) & (m['diff_speech_wer'] < 0)].orig_id


for orig_id in hif1_lowe_sp:
    print(orig_id)
    print("GOLD:         ", speech_asr_df[speech_asr_df.orig_id == orig_id].orig_sent.values[0])
    print("1-BEST:       ", speech_asr_df[speech_asr_df.orig_id == orig_id].asr_sent.values[0])
    print("SPEECH RANKED:", pair_speech_df[pair_speech_df.orig_id == orig_id].asr_sent.values[0])
    print("TEXT RANKED:  ", pair_text_df[pair_text_df.orig_id == orig_id].asr_sent.values[0])
    print()




sorted_f1 = m[(m['diff_speech_f1'] > 0) & (m['diff_speech_wer'] > 0)].sort_values('diff_speech_f1')[['diff_speech_wer','diff_speech_f1', 'orig_id']]


