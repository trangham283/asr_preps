import os
import re
import sys
import glob
import pandas as pd
import argparse
from collections import defaultdict
import numpy as np
import json

def make_array(frames):
    return np.array(frames).T

def convert_to_array(str_vector):
    str_vec = str_vector.replace('[','').replace(']','').replace(',','').split()
    num_list = []
    for x in str_vec:
        x = x.strip()
        if x != 'None': num_list.append(float(x))
        else: num_list.append(np.nan)
    return num_list

def has_bad_alignment(num_list):
    for i in num_list:
        if i < 0 or np.isnan(i): return True
    return False

def find_bad_alignment(num_list):
    bad_align = []
    for i in range(len(num_list)):
        if num_list[i] < 0 or np.isnan(num_list[i]): 
            bad_align.append(i)
    return bad_align

def check_valid(num):
    if num < 0 or np.isnan(num): return False
    return True

def clean_up_old(stimes, etimes):
    if not check_valid(stimes[-1]):
        stimes[-1] = max(etimes[-1] - num_sec, 0)
    
    if not check_valid(etimes[0]):
        etimes[0] = stimes[0] +  num_sec
    
    for i in range(1,len(stimes)-1):
        this_st = stimes[i]
        prev_st = stimes[i-1]
        next_st = stimes[i+1]

        this_et = etimes[i]
        prev_et = etimes[i-1]
        next_et = etimes[i+1]   
   
        if not check_valid(this_st) and check_valid(prev_et):
            stimes[i] = prev_et

        if not check_valid(this_st) and check_valid(prev_st):
            stimes[i] = prev_st + num_sec

    for i in range(1,len(etimes)-1)[::-1]:
        this_st = stimes[i]
        prev_st = stimes[i-1]
        next_st = stimes[i+1]

        this_et = etimes[i]
        prev_et = etimes[i-1]
        next_et = etimes[i+1]   
        if not check_valid(this_et) and check_valid(next_st):
            etimes[i] = next_st

        if not check_valid(this_et) and check_valid(next_et):
            etimes[i] = next_et - num_sec

    return stimes, etimes

def clean_up(stimes, etimes, tokens, dur_stats):
    total_raw_time = etimes[-1] - stimes[0]
    total_mean_time = sum([dur_stats[w]['mean'] for w in tokens])
    scale = min(total_raw_time / total_mean_time, 1)

    no_start_idx = find_bad_alignment(stimes)
    no_end_idx = find_bad_alignment(etimes)

    # fix start times first
    for idx in no_start_idx:
        if idx not in no_end_idx:
            # this means the word does have an end time; let's use it
            stimes[idx] = etimes[idx] - scale*dur_stats[tokens[idx]]['mean']
        else:
            # this means the idx does not s/e times -- just use prev's start
            stimes[idx] = stimes[idx-1] 

    # now all start times should be there
    for idx in no_end_idx:
        etimes[idx] = stimes[idx] + scale*dur_stats[tokens[idx]]['mean']
            
    return stimes, etimes

# NaN cases are usually for contractions
# -1 are usually for missed/inserted words
# if end time is NaN: use end time of next word
# if start time is NaN: use start time of prev word
def get_duration_stats(train_file):
    df = pd.read_csv(train_file, sep="\t")
    dur_dict = {}
    for i, row in df.iterrows():
        tokens = row.sentence.strip().split()
        stimes = convert_to_array(row.start_times)
        etimes = convert_to_array(row.end_times)
        # left-right pass, to fix start times
        for j in range(1, len(tokens)-1):
            tok, stime = tokens[j], stimes[j]
            if stime < 0:
                continue
            if np.isnan(stime):
                stime = stimes[j-1]
                stimes[j] = stime
        # right-left pass, to fix end times
        for j in range(1, len(tokens)-1)[::-1]:
            tok, etime = tokens[j], etimes[j]
            if etime < 0:
                continue
            if np.isnan(etime): 
                etime = etimes[j+1]
                etimes[j] = etime 
        tok, stime, etime = tokens[0], stimes[0], etimes[0]
        if np.isnan(stime) or stime < 0 or etime < 0:
            continue
        if np.isnan(etime): 
            etime = etimes[1]
            etimes[0] = etime
        word_dur = etime - stime
        if np.isnan(word_dur): print(i, j, stime, etime, tok)
        if tok not in dur_dict:
            dur_dict[tok] = []
        dur_dict[tok].append(word_dur)
        for j in range(1, len(tokens)-1):
            tok, stime, etime = tokens[j], stimes[j], etimes[j]
            if stime < 0 or etime < 0:
                continue
            word_dur = etime - stime
            if np.isnan(word_dur): print(i, j, etime, stime, tok)
            if tok not in dur_dict:
                dur_dict[tok] = []
            dur_dict[tok].append(word_dur)
        tok, stime, etime = tokens[-1], stimes[-1], etimes[-1]
        if np.isnan(etime) or etime < 0 or stime < 0:
            continue
        if np.isnan(stime):
            stime = stimes[-2]
            stimes[-1] = stime
        word_dur = etime - stime
        if np.isnan(word_dur): print(i, j, stime, etime, tok)
        if tok not in dur_dict:
            dur_dict[tok] = []
        dur_dict[tok].append(word_dur)
    return dur_dict

def get_time_boundaries(split, feat_types):
    data_file = os.path.join(time_dir, split + '_mrg.tsv')
    dur_stats_file = os.path.join(data_dir, 'avg_word_stats.pickle')
    dur_stats = pickle.load(open(dur_stats_file))

    df = pd.read_csv(data_file, sep='\t')
    sw_files = set(df.file_id.values)
    for sw in sw_files:
        this_dict = defaultdict(dict) 
        for speaker in ['A', 'B']:
            this_df = df[(df.file_id==sw)&(df.speaker==speaker)]
            for i, row in this_df.iterrows():
                tokens = row.sentence.strip().split()
                stimes = convert_to_array(row.start_times)
                etimes = convert_to_array(row.end_times)
                
                if len(stimes)==1: 
                    if (not check_valid(stimes[0])) and (not check_valid(etimes[0])):
                        print("no time available for sentence", row.sent_id)
                        continue
                    elif not check_valid(stimes[0]):
                        stimes[0] = max(etimes[0] - dur_stats[tokens[0]]['mean'], 0)
                    else:
                        etimes[0] = stimes[0] + dur_stats[tokens[0]]['mean']
                 
                if check_valid(stimes[0]): 
                    begin = stimes[0]
                else:
                    # cases where the first word is unaligned
                    if check_valid(etimes[0]): 
                        begin = max(etimes[0] - dur_stats[tokens[0]]['mean'], 0) 
                        stimes[0] = begin
                    elif check_valid(stimes[1]):
                        begin = max(stimes[1] - dur_stats[tokens[-1]]['mean'], 0)
                        stimes[0] = begin
                    else:
                        continue

                if check_valid(etimes[-1]): 
                    end = etimes[-1]
                else:
                    # cases where the last word is unaligned
                    if check_valid(stimes[-1]): 
                        end = stimes[-1] + dur_stats[tokens[-1]]['mean']
                        etimes[-1] = end
                    elif check_valid(etimes[-2]):
                        end = etimes[-2] + dur_stats[tokens[-1]]['mean']
                        etimes[-1] = end
                    else:
                        continue
                
                # final clean up
                stimes, etimes = clean_up(stimes, etimes, tokens, dur_stats)
                assert len(stimes) == len(etimes) == len(tokens)

                sframes = [int(np.floor(x*100)) for x in stimes]
                eframes = [int(np.ceil(x*100)) for x in etimes]
                s_frame = sframes[0]
                e_frame = eframes[-1]
                word_lengths = [e-s for s,e in zip(sframes,eframes)]
                invalid = [x for x in word_lengths if x <=0]
                toolong = [x for x in word_lengths if x >=100]
                if len(invalid)>0: 
                    print("End time < start time for: ",row.sent_id,row.speaker)
                    print(invalid)
                    continue
                if len(toolong)>0:
                    item = "Too long: " + row.sent_id + ' ' +row.speaker +' ' + str(toolong) + '\n'
                    print(item)

                offset = s_frame
                word_bounds = [(x-offset,y-offset) for x,y in zip(sframes, eframes)]
                assert len(word_bounds) == len(tokens)
                globID = row.sent_id.replace('~','_'+speaker+'_')
                
                this_dict[globID]['windices'] = word_bounds
                this_dict[globID]['word_dur'] = [etimes[i]-stimes[i] for i in range(len(stimes))]
        dict_name = os.path.join(output_dir, split, sw + '_prosody.pickle')
        pickle.dump(this_dict, open(dict_name, 'w'))
    return

def norm_energy_by_turn(this_data):
    feat_dim = 41
    turnA = np.empty((feat_dim,0)) 
    turnB = np.empty((feat_dim,0))
    for k in this_data.keys():
        fbank = this_data[k]['fbank']
        fbank = np.array(fbank).T
        if 'A' in k:
            turnA = np.hstack([turnA, fbank])
        else:
            turnB = np.hstack([turnB, fbank])
    meanA = np.mean(turnA, 1) 
    stdA = np.std(turnA, 1)
    meanB = np.mean(turnB, 1)
    stdB = np.std(turnB, 1)
    maxA = np.max(turnA, 1)
    maxB = np.max(turnB, 1)
    return meanA, stdA, meanB, stdB, maxA, maxB 

def process_data_both(data_dir, split, sent_vocab, parse_vocab, normalize=False):
    data_set = [[] for _ in _buckets]
    sentID_set = [[] for _ in _buckets]
    dur_stats_file = os.path.join(data_dir, 'avg_word_stats.pickle')
    dur_stats = pickle.load(open(dur_stats_file))
    global_mean = np.mean([x['mean'] for x in dur_stats.values()])
    split_path = os.path.join(data_dir, split)
    split_files = glob.glob(split_path + "/*")
    for file_path in split_files:
        this_data = pickle.load(open(file_path))

        if normalize:
            meanA, stdA, meanB, stdB, maxA, maxB  = norm_energy_by_turn(this_data)

        for k in this_data.keys():
            sentence = this_data[k]['sents']
            parse = this_data[k]['parse']
            windices = this_data[k]['windices']
            pause_bef = this_data[k]['pause_bef']
            pause_aft = this_data[k]['pause_aft']

            # features needing normalization
            word_dur = this_data[k]['word_dur']
            pitch3 = make_array(this_data[k]['pitch3'])
            fbank = make_array(this_data[k]['fbank'])
            if normalize:
                exp_fbank = np.exp(fbank)
                # normalize energy by z-scoring
                if 'A' in k:
                    mu = meanA
                    sigma = stdA
                    hi = maxA
                else:
                    mu = meanB
                    sigma = stdB
                    hi = maxB

                #e_total = np.sum(mu[1:])
                #e0 = (fbank[0, :] - mu[0]) / sigma[0]
                #elow = np.sum(fbank[1:21,:],0)/e_total
                #ehigh = np.sum(fbank[21:,:],0)/e_total

                e_total = exp_fbank[0, :]
                e0 = fbank[0, :] / hi[0]
                elow = np.log(np.sum(exp_fbank[1:21,:],0)/e_total)
                ehigh = np.log(np.sum(exp_fbank[21:,:],0)/e_total)

                energy = np.array([e0,elow,ehigh])

                # normalize word durations by dividing by mean
                words = sentence.split()
                assert len(word_dur) == len(words)
                for i in range(len(words)):
                    if words[i] not in dur_stats:
                        print("No mean dur info for word ", words[i])
                        wmean = global_mean
                    wmean = dur_stats[words[i]]['mean']
                    # clip at 5.0
                    word_dur[i] = min(word_dur[i]/wmean, 5.0)
            else:
                energy = fbank[0,:].reshape((1,fbank.shape[1]))

            pitch3_energy = np.vstack([pitch3, energy])

    return data_set, sentID_set

            
hop = 10.0 # in msec
num_sec = 0.04 # amount of time to approximate extra frames when no time info available
def main():
    pa = argparse.ArgumentParser(
        description='Preprocess/reprocess some parsing data timing info')
    pa.add_argument('--split', help='data split', default='dev')
    pa.add_argument('--step', help='get_dur_stats, correct_times, ???')
    pa.add_argument('--train_file', 
        help='training file with word-level time data',
        default='/g/ssli/data/CTS-English/swbd_align/swbd_trees/train2_mrg.tsv')
    args = pa.parse_args()
    # debug
    split = args.split
    step = args.step
    train_file = args.train_file
    
    if step == 'get_dur_stats':
        # STEP 1: get duration statistics
        stats = {}
        dur_stats = get_duration_stats(train_file)
        for word, times in dur_stats.items():
            stats[word] = {}
            stats[word]['count'] = len(times)
            stats[word]['mean'] = np.mean(times)
            stats[word]['std'] = np.std(times)
        outfile = 'avg_word_stats.json'
        with open(outfile, 'w') as fout:
            json.dump(stats, fout, indent=2)
    else:
        # not yet implemented
        print("")
    exit(0)


    # normalize and process data into buckets
    #normalize = True
    #this_set, sentID_set = process_data_both(output_dir, split, sent_vocab, parse_vocab, normalize)
    #this_file = os.path.join(output_dir, split + '_prosody_normed.pickle')
    #pickle.dump(this_set, open(this_file,'w'))
    #sent_file = os.path.join(output_dir, split + '_sentID.pickle')
    #pickle.dump(sentID_set, open(sent_file, 'w'))

if __name__ == "__main__":
    main()


