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

--------------------------------------
## OLD STUDD

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



