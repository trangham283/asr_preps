#!/usr/bin/env python3

import random
from utils import *
random.seed(0)


dep_type = 'labeled'
asr_dir = "asr_output/nbest/for_parsing"
add_edit = 1
criteria = 'bracket'
prefix = 'overall' if criteria=='dependency' else 'bracket'
classifier = 'LR'
features = 'fl2'
nsamples = 5

model_text = "exp_medians/{}_{}_{}_{}_dep-{}_edit-{}_{}_spls.pkl".format(
        classifier, 1700, features, 
        criteria, dep_type, add_edit, nsamples)

model_speech = "exp_medians/{}_{}_{}_{}_dep-{}_edit-{}_{}_spls.pkl".format(
        classifier, 3700, features, 
        criteria, dep_type, add_edit, nsamples)

model_pkl = model_speech
feats = pkl2feats(model_pkl)

if bool(add_edit):
    add_edit_str = "_unedit.pickle"
else:
    add_edit_str = ".pickle"

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

# Correlation with wer (all, including 10-hyp)
print("Correlation on the whole set (inclde all 10 hypotheses)")
r_sp = stats.spearmanr(speech_df.bracket_f1, speech_df.wer)
r_tt = stats.spearmanr(text_df.bracket_f1, text_df.wer)
print("Text, Speech:", r_tt, r_sp)

compute_oracles(text_df, dep=False)
compute_oracles(speech_df, dep=False)


with open(model_text, 'rb') as f:
    clf_text = pickle.load(f)

with open(model_speech, 'rb') as f:
    clf_speech = pickle.load(f)

test_df = text_df
clf = clf_text
point_df, point_ff, point_wer = get_res(test_df, clf, feats, prefix) 
pair_df, pair_ff, pair_wer = pred_by_pair(test_df, clf, feats, prefix)
r_point = stats.spearmanr(point_df[prefix+'_f1'], point_df.wer)
r_pair = stats.spearmanr(pair_df[prefix+'_f1'], pair_df.wer)
print(r_point, r_pair)

test_df = speech_df
clf = clf_speech
point_df, point_ff, point_wer = get_res(test_df, clf, feats, prefix) 
pair_df, pair_ff, pair_wer = pred_by_pair(test_df, clf, feats, prefix)
r_point = stats.spearmanr(point_df[prefix+'_f1'], point_df.wer)
r_pair = stats.spearmanr(pair_df[prefix+'_f1'], pair_df.wer)
print(r_point, r_pair)





