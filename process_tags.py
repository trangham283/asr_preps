#!/usr/bin/env python3

import os
import argparse
import pandas as pd
import glob
import numpy as np
import json


def process_tags(args):
    asr_dir = args.asr_dir
    out_name = args.out_name
    tsv_file = args.tsv_file
    tag_file = args.tag_file

    df = pd.read_csv(tsv_file, sep="\t")
    sent_tags = open(tag_file).readlines()
    sent_tags = [x.strip().split() for x in sent_tags]
    num_tags = [len(x) for x in sent_tags]
    orig_sents = df.asr_sent.tolist()
    num_toks = [len(x.split()) for x in orig_sents]
    comb = np.array([num_toks, num_tags])
    diff_toks = comb[0, :] - comb[1, :]
    if sum(diff_toks) != 0:
        idx = np.where(diff_toks !=0)[0]
        for i in idx:
            #print(orig_sents[i], sent_tags[i])
            print(orig_sents[i])
    else:
        with open(out_name, 'w') as f:
            for x in sent_tags:
                item = " ".join(x)
                f.write("{}\n".format(item))
    return


def main():
    """main function"""
    pa = argparse.ArgumentParser(
            description='Get POS tags and check/process them')
    pa.add_argument('--asr_dir', help='asr output dir', 
            default='asr_output/nbest')
    pa.add_argument('--out_name', help='output dir', 
            default='asr_output/nbest/dev_asr_sent_with_tags.txt')
    pa.add_argument('--tsv_file', help='tsv file',
            default='asr_output/nbest/dev_asr_pa_nbest.tsv')
    pa.add_argument('--tag_file', help='tag file',
            default='asr_output/nbest/dev_asr_sents.tags')
    args = pa.parse_args()
    
    process_tags(args)

    
    exit(0)


if __name__ == '__main__':
    main()

