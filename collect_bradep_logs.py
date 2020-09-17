#!/usr/bin/env python3
import argparse, os, glob
import pandas as pd 
import numpy as np

columns = ['sent_num', 
'bracket_match', 'bracket_gold', 'bracket_test', 'bracket_cross', 
'overall_match', 'overall_gold', 'overall_test', 
'open_match', 'open_gold', 'open_test']



def collect_results(data_dir, split, model, dep_type, unedit=True):
    list_row = []
    if unedit:
        prefix = os.path.join(data_dir, 
                "{}_{}_bradep_{}_unedit_*.log".format(split, dep_type, model))
    else:
        prefix = os.path.join(data_dir, 
                "{}_{}_bradep_{}_*.log".format(split, dep_type, model))
    files = glob.glob(prefix)
    for filename in files:
        sent_num = int(filename.split("_")[-1][:-4]) + 1
        ll = open(filename).readlines()
        scores = ll[4].split()
        scores = [int(x) for x in scores]
        scores[0] = sent_num
        row = dict(zip(columns, scores))
        list_row.append(row)
    df = pd.DataFrame(list_row)
    return df

#data_dir = "/s0/ttmt001/sparseval_aligns/logs"
#out_dir = "asr_output/nbest/for_parsing"

def main():
    """main function"""
    pa = argparse.ArgumentParser(
            description='Collect log results')
    pa.add_argument('--data_dir', help='data dir', 
            default='/s0/ttmt001/sparseval_aligns/logs')
    pa.add_argument('--out_dir', help='out dir', 
            default='asr_output/nbest/for_parsing')
    pa.add_argument('--split', help='data split', default='dev')
    pa.add_argument('--model', help='model ID', type=int)
    pa.add_argument('--dep_type', help='dep type', default='unlabeled')
    pa.add_argument('--unedit', help='include EDIT in evaluation or not', default=0, type=int)
    args = pa.parse_args()
    
    data_dir = args.data_dir
    split = args.split
    model = args.model
    unedit = bool(args.unedit)
    dep_type = args.dep_type
    out_dir = args.out_dir

    if unedit:
        outname = os.path.join(out_dir, "{}_{}_bradep_{}_unedit.tsv".format(split, dep_type, model))
    else:
        outname = os.path.join(out_dir, "{}_{}_bradep_{}.tsv".format(split, dep_type, model))

    df = collect_results(data_dir, split, model, dep_type)
    df.to_csv(outname, sep="\t", index=False)

    exit(0)

if __name__ == '__main__':
    main()

