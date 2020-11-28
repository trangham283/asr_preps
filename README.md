# Misc codes for ASR-parsing experiments

Mostly for backing up code

## UPDATE:
* added extra 3 frames before/after each turn to avoid noisy cut offs

## Steps
### Preprocessing

* Compute average duration stats: in `get_utterance_times.py`, function `write_dur_stats()`

* Write commands to split channels in audio files
`python get_utterance_times.py --split {dev,test} --step split --task {da,parse}`
* Write commands to trim on uttereance level
`python get_utterance_times.py --split {dev,test} --step trim --task {da,parse}`

* Write commands to trim on turn level (da only):
`python get_utterance_times.py --split {dev,test} --step turns --task da`

* Run the corresponding commands `cmd*.sh`

### Decoding 
* Kaldi decoding script: `./decode_audio_$task_$split`
* NOTE: `3942_B_0126` is too short, kaldi could not extract acoustic frames from it, so skip
* run acronym map: `./run_acronym_map.sh`
* Combine info from asr outputs:
`python parse_asr_output.py --split {dev,test} --task {pa,da}`


## FOR PARSING
* Prepare sentences for tagging and then parsing
```
cut -f6 test_asr_pa_nbest.tsv > test_asr_sents.txt
cut -f9 test_asr_pa_nbest.tsv > test_asr_mrg.txt 
cut -f1 test_asr_pa_nbest.tsv > test_asr_sent_ids.txt 

cut -f6 dev_asr_pa_nbest.tsv > dev_asr_sents.txt
cut -f9 dev_asr_pa_nbest.tsv > dev_asr_mrg.txt 
cut -f1 dev_asr_pa_nbest.tsv > dev_asr_sent_ids.txt 
```

* NOTE:
"y'all" doesn't get tagged correctly if I pre-tokenize; ended manually put back to "y'all" before running the tagger.

* then remove first header row

```
./run_tagger.sh

python process_tags.py --out_name asr_output/nbest/test_asr_sent_with_tags.txt --tsv_file asr_output/nbest/test_asr_pa_nbest.tsv --tag_file asr_output/nbest/test_asr_sents.tags
```

* then fix tags associated with "i" in `dev_asr_sent_with_tags.txt`
in vim:
```
%s/i_FW/i_PRP/g
%s/i_LS/i_PRP/g
%s/i_PRP b_NN/i_NN b_NN/g
%s/i_PRP t_NN/i_NN t_NN/g
```

* in python2 environment (bc of pickle module mostly): `python get_speech_features.py --split {dev, test}`

* then, in ../self-attentive-parser: `./run_parse_asr.sh`

* then add "TOP" nodes
in vim, `*.parse`:
```
%s/$/)/g
%s/^/(S /g
```

* run sparseval: `./run_sparseval.sh`

## FOR DA 
* Make turn-level cmd trim (done): `python get_utterance_times.py --split {dev,test} --step turns --task da` 

* Run kaldi decoding: `./decode_audio.sh da {dev,test} 10` 

* Prepare for writing to json files like in the joint-seg-da directory:
`python get_utterance_times.py --split {dev,test} --step write_df`

* 

--------------------------------------
## OLD STUFF and Error notes

For ASR WER computation, I installed this package:
`pip install jiwer`
(in envs/py3.6-transformers-cpu)

python compute_wer.py --task da --split test
For set test, in task da:
    29304 words in ref; 27763 in asr; 4069 utterances total
    WER = 0.24116161616161616
python compute_wer.py --task da --split dev
For set dev, in task da:
    25298 words in ref; 23598 in asr; 3265 utterances total
    WER = 0.22523519645821805

python compute_wer.py --task parse --split dev
For set dev, in task parse:
    48092 words in ref; 45337 in asr; 5670 utterances total
    WER = 0.1917574648590202
python compute_wer.py --task parse --split test
For set test, in task parse:
    46436 words in ref; 43901 in asr; 5793 utterances total
    WER = 0.20057713842708244

Errors in time alignements:
Dev set
In my processing, not in prev. parsing:
{}

In prev parsing, not in mine: (9 utterancs)
{'sw4617_A_0097', 'sw4617_B_0108', 'sw4565_A_0107', 'sw4792_A_0022', 'sw4928_A_0014', 'sw4890_B_0048', 'sw4928_A_0013', 'sw4785_A_0006', 'sw4858_B_0069'}

Test set
In my processing, not in prev. parsing:
{}

In prev parsing, not in mine: (9 utterancs)
{'sw4104_B_0098', 'sw4064_A_0040', 'sw4019_A_0159', 'sw4149_A_0108', 'sw4104_A_0073', 'sw4127_A_0110', 'sw4150_A_0031', 'sw4099_A_0173', 'sw4130_B_0084'}

## Check extract_ta_features.py in prosodic-anomalies

dev-set: no parse speech feats: (51)
```
sw4519_A_0041
sw4519_A_0072
sw4519_B_0054
sw4548_A_0023
sw4548_A_0049
sw4565_A_0107
sw4565_A_0109
sw4565_B_0004
sw4565_B_0085
sw4565_B_0095
sw4565_B_0100
sw4572_B_0014
sw4603_B_0061
sw4611_B_0021
sw4617_A_0097
sw4617_B_0107
sw4617_B_0108
sw4626_B_0115
sw4633_A_0102
sw4633_B_0012
sw4649_B_0054
sw4659_A_0032
sw4659_B_0082
sw4682_A_0009
sw4682_A_0096
sw4707_B_0026
sw4720_A_0018
sw4725_A_0032
sw4725_B_0030
sw4728_B_0015
sw4736_B_0052
sw4785_A_0006
sw4792_A_0022
sw4796_A_0053
sw4796_A_0059
sw4796_A_0063
sw4796_B_0074
sw4796_B_0076
sw4796_B_0078
sw4796_B_0082
sw4796_B_0085
sw4812_A_0117
sw4858_B_0069
sw4886_A_0025
sw4890_B_0020
sw4890_B_0048
sw4890_B_0068
sw4928_A_0004
sw4928_A_0013
sw4928_A_0014
sw4936_B_0034
```

test-set: no-parse speech feats: (45)
```
sw4019_A_0159
sw4026_A_0040
sw4038_B_0051
sw4051_A_0037
sw4055_A_0106
sw4056_A_0070
sw4056_A_0076
sw4056_A_0079
sw4056_A_0083
sw4056_A_0114
sw4056_B_0097
sw4064_A_0039
sw4064_A_0040
sw4071_A_0020
sw4071_B_0030
sw4079_B_0006
sw4080_A_0086
sw4080_A_0120
sw4082_A_0012
sw4082_A_0087
sw4090_A_0016
sw4090_A_0035
sw4092_A_0033
sw4092_A_0052
sw4092_A_0154
sw4099_A_0031
sw4099_A_0173
sw4103_A_0035
sw4104_A_0073
sw4104_B_0093
sw4104_B_0098
sw4108_A_0118
sw4109_B_0078
sw4113_A_0083
sw4114_B_0037
sw4114_B_0071
sw4114_B_0123
sw4127_A_0047
sw4127_A_0110
sw4130_B_0084
sw4133_B_0069
sw4138_B_0079
sw4149_A_0108
sw4150_A_0029
sw4150_A_0031
```

