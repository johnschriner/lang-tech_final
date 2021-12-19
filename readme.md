The data is part of a set of scrapes and normalizations done by the WikiPron team.<br />
The data for Adyghe contains 4,895 graphemes and their phonemes.<br />
4.45% of the words from the scrape were filtered for normalization and consistent transcription; this was done by creating a phonelist that would be used to remove non-native segments (e.g. the velar fricative from _Bach_, pronounced /bɑːx/ while /x/ is absent in modern English and is excluded)[^1].<br />
I will be using FairSeq to create an encoder-decoder LSTM for Adyge. <br />

### Split data
```
INFO: Total set:	4,895 lines
INFO: Train set:	3,916 lines
INFO: Development set:	489 lines
INFO: Test set:		490 lines
```
### Preprocessing the split data for FairSeq
```
import contextlib
import csv

# data was shuffled using `shuf` and split 80-10-10 using `split.py`

TRAIN = "ady_train.tsv"
TRAIN_G = "train.ady.g"
TRAIN_P = "train.ady.p"

DEV = "ady_dev.tsv"
DEV_G = "dev.ady.g"
DEV_P = "dev.ady.p"

TEST = "ady_test.tsv"
TEST_G = "test.ady.g"
TEST_P = "test.ady.p"

with contextlib.ExitStack() as stack:
    source = csv.reader(stack.enter_context(open(TRAIN, "r")), delimiter="\t")
    g = stack.enter_context(open(TRAIN_G, "w"))
    p = stack.enter_context(open(TRAIN_P, "w"))
    for graphemes, phones in source:
        print(" ".join(graphemes), file=g)
        print(phones, file=p)
# Processes development data.
with contextlib.ExitStack() as stack:
    source = csv.reader(stack.enter_context(open(DEV, "r")), delimiter="\t")
    g = stack.enter_context(open(DEV_G, "w"))
    p = stack.enter_context(open(DEV_P, "w"))
    for graphemes, phones in source:
        print(" ".join(graphemes), file=g)
        print(phones, file=p)
    # Processes test data.
with contextlib.ExitStack() as stack:
    source = csv.reader(stack.enter_context(open(TEST, "r")), delimiter="\t")
    g = stack.enter_context(open(TEST_G, "w"))
    p = stack.enter_context(open(TEST_P, "w"))
    for graphemes, phones in source:
        print(" ".join(graphemes), file=g)
        print(phones, file=p)
```

### Preprocessing in FairSeq:<br />
```
fairseq-preprocess \
    --source-lang ady.g \
    --target-lang ady.p \
    --trainpref train \
    --validpref dev \
    --testpref test \
    --tokenizer space \
    --thresholdsrc 2 \
    --thresholdtgt 2
```     
### Training in FairSeq:<br />
```
fairseq-train \
    data-bin \
    --source-lang ady.g \
    --target-lang ady.p \
    --encoder-bidirectional \
    --seed 230 \
    --arch lstm \
    --dropout 0.2 \
    --lr .001 \
    --max-update 800\
    --no-epoch-checkpoints \
    --batch-size 50 \
    --clip-norm 1 \
    --label-smoothing .1 \
    --optimizer adam \
    --clip-norm 1 \
    --criterion label_smoothed_cross_entropy\
    --encoder-embed-dim 512 \
    --decoder-embed-dim 512 \
    --encoder-layers 4 \
    --decoder-layers 4 \
 ```

### Evaluation
```
fairseq-generate \
    data-bin \
    --source-lang ady.g \
    --target-lang ady.p \
    --path checkpoints/checkpoint_best.pt \
    --gen-subset valid \
    --beam 8 \
    > predictions.txt
./wer.py predictions.txt
```
We will tweak `dev` for model selection and the best model will be used on `test` data. <br />

Parameters above on the `dev` data:<br />
`WER:	29.80`<br />
halving the hidden layers from 4 to 2 and halving their size from 512 to 256: <br />
`WER:    27.55`<br />
Continuing and doubling the batch size from 50 to 100:<br />
`WER:	25.71`<br />
<br />
Continuing and increasing the batch size from 100 to 500:<br />
```
fairseq-train \
    data-bin \
    --source-lang ady.g \
    --target-lang ady.p \
    --encoder-bidirectional \
    --seed 230 \
    --arch lstm \
    --dropout 0.2 \
    --lr .001 \
    --max-update 800\
    --no-epoch-checkpoints \
    --batch-size 500 \
    --clip-norm 1 \
    --label-smoothing .1 \
    --optimizer adam \
    --clip-norm 1 \
    --criterion label_smoothed_cross_entropy\
    --encoder-embed-dim 256 \
    --decoder-embed-dim 256 \
    --encoder-layers 2 \
    --decoder-layers 2 \
```
This resulted in a longer processing time, more epochs, but a better lower loss value and WER: <br />
`WER:	24.34` <br />

Choosing this model, we can now run it on the `test` data: <br />
`WER:	26.73`<br />

### A close examination of hypotheses and phone errors

|Target |Hypothesis | Phone error notes |
|------|------------|-----------|
|ħ aː l ʐʷ a ʁʷ aː n|ħ aː l ʒʷ a ʁʷ aː n| The source is [хьалжъогъуан] - Only instance[^2] of /ʐʷ/ in the test data, and 5 instances of /ʒʷ/|
|b ʁ a kʲʼ a ħ|b ʁ a t͡ʃʼ a ħ|The source is бгъэкӏэхь - there are 14 instances of /kʲʼ/ and 39 instances of /t͡ʃʼ/. /t͡ʃʼ/ occurs quite often so correcting this error would vastly improve the WER.|
|ʂʷ a t a j ħ a t a j t͡ʃ|ʃʷ a t a j ħ a t a j t͡ʃʼ|Several occurrences missing or added ejective to the voiceless postalveolar affricate|

### Variable phones for [шъ]
|Target |Hypothesis | Phone error notes |
|------|------------|-----------|
|n aː ʂʷ χʷ a|n aː ʃʷ χʷ a|The source is [нашъухъо] - /ʂʷ/ and /ʃʷ/ are numerous in the data|
|kʷ ə ɬ a ʃʷ|kʷ ə ɬ a ʂʷ|The opposite of the example directly above - these sounds are phonetically very similar|
|n a ʂʷ ə|n a ʃʷ| The grapheme is [нэшъу]|
|ʃʷʼ a ʃʷʼ χʷ ə n ʁ a|ʃʷʼ a ʂ χʷ ə n ə ʁ a| /ʂ/ is often used to represent [шъ] |
|ʂʷ a t a j ħ a t a j t͡ʃ|ʃʷ a t a j ħ a t a j t͡ʃʼ|[шъотехьэтекӏ] - [шъ] and [ш] in Adyghe seem to have several possible phones|
|ʃʷʼ aː m pʼ a|ʃʷ aː m pʼ a| The grapheme is [шъуампӏэ] and correctly predicted.  Another instance of the unpredictability of /шъ/|

### Glottal stop
|Target |Hypothesis | Phone error notes |
|------|------------|-----------|
|ʔ aː q ə l |aː q ə l| The target contains a glottal stop that wasn't predicted.  Training data had the same root in [акъылыгъу] that does _not_ have the glottal stop |
|ʔ aː ʃ a|ʔ aː ʃ a| Correctly predicted a glottal stop when at the beginning on the word.  This is the only 1 instance (of 8 where the word begins with [аш]) in all of the WikiPron data and the LSTM correctly predicted a glottal stop to precede the long /a/ word-initially|

### [л] and [г] variation
|Target |Hypothesis | Phone error notes |
|------|------------|-----------|
|z aː w l a|z aː w ɮ a|The graph [л] is incorrectly hypothesized to be /ɮ/|
|ʔ aː ɮ ə| ʔ aː l ə| The grapheme [ӏалы] again contains [л] and incorrectly predicts /l/|
|p ə l|p ə ɮ| Another occurence|
|ɬ a p s|ɬ a p s|From the source grapheme [лъэпс] the [л] is correctly predicted as /ɬ/|
|ʁ a f a b aː pʼ a| ʁ a f a b aː pʼ a|The grapheme is [гъэфэбапӏэ], the [г] here pronounced as /ʁ/ whereas in Russian, for example the graph [г] is pronounced as /g/ and sometimes at the end of words as /k/ or /v/.  In the data, [г] is represented as /ʁ/, /ʁʷ/, /ɡʷ/, or /ɣ/|

### Vowels
|Target |Hypothesis | Phone error notes |
|------|------------|-----------|
|m a r aː kʷʼ aː t͡sʼ a|m a r aː kʷʼ aː p t͡sʼ a| successfully predicted long vowels throughout the experiment|
|w ə ɡʲ|w ə ɡʲ| We see that [y] is represented phonetically as /wə/| 
|w f a n| w ə f a n |The grapheme [уфэн] in this case does _not_ have the schwa /ə/|
|z ə p a ʁʷ a t ħ a ʑ ə n | z ə p a ʁʷ ə t ħ a ʑ ə n | Incorrectly hypothesized a schwa /ə/|
|x ə ʁ a χʷ ə n a n ə qʷ ə| ə ʁ a χʷ ə n a n ə qʷ| Word-final schwa omitted in hypothesis.  The source grapheme is [хыгъэхъунэныкъу].  _n.b._ [ы] is correctly mapped to the schwa /ə/ throughout the experiment.<br />|

<br />
The challenge we find is:<br />
A single grapheme can be mapped to several different phonemes.<br />
There has been some discussion of how the Cyrillic script cannot properly be used to represent the consonantal inventory of the language.<br />
Here we find evidence through our experiments that this may be the case.<br />

## Transformer
Trying now the state-of-the-art Transformer grapheme to phoneme conversion model using the Moses tokenizer (otherwise _vanilla_ in its parameters):
```
fairseq-train data-bin 
    --source-lang ady.g 
    --target-lang ady.p     
    --clip-norm 0.1 
    --dropout 0.2 
    --max-tokens 2048     
    --optimizer adam --adam-betas '(0.9, 0.98)' 
    --clip-norm 0.0     
    --lr 5e-4 
    --lr-scheduler inverse_sqrt     
    --criterion label_smoothed_cross_entropy     
    --update-freq 4     
    --log-format json     
    --keep-interval-updates 100 
    --save-interval-updates 3000  
    --log-interval 50     
    --arch transformer 
    --save-dir checkpoints/transformer 
    --no-epoch-checkpoints  
    --tokenizer moses
```

```
fairseq-generate \
    data-bin \
    --source-lang ady.g \
    --target-lang ady.p \
    --path checkpoints/transformer/checkpoint_best.pt \
    --gen-subset test \
    --beam 8 \
    > predictions.txt
```
```
WER:	85.71
```
The high WER is due in large part to not predicting the correct phoneme due to ambiguity and incorrect mapping. <br />

As mentioned in the paper, moving beyond the vanilla parameters for Transformer would likely provide results that could equal or overtake the LSTM results above.  The authors of the Transformer paper had significant processing time and resources: their computation ran on 8 GPUs for 3.5 days.
<br />

[^1]: Ashby, L. F. E., Bartley, T. M., Clematide, S., Del Signore, L., Gibson, C., Gorman, K., Lee-Sikka, Y., Makarov, P., Malanoski, A., Miller, S., Ortiz, O., Raff, R., Sengupta, A., Seo, B., Spektor, Y., & Yan, W. (2021). Results of the second sigmorphon shared task on multilingual grapheme-to-phoneme conversion. _Proceedings of the 18th SIGMORPHON Workshop on Computational Research in Phonetics, Phonology, and Morphology_, 115–125. https://doi.org/10.18653/v1/2021.sigmorphon-1.13
[^2]: by 'instance' I mean: occurred in the Target field; no other fields were counted.
