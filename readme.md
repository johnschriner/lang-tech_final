The data is part of a set of scrapes and normalizations done by the WikiPron team.<br />
The data for Adyghe contains 4,895 graphemes and their phonemes.<br />
4.45% of the words from the scrape were filtered due to normalization, _______ <br />
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
    --seed 21 \
    --arch lstm \
    --encoder-bidirectional \
    --dropout .2 \
    --encoder-embed-dim 128 \
    --decoder-embed-dim 128 \
    --decoder-out-embed-dim 128 \
    --encoder-hidden-size 512 \
    --decoder-hidden-size 512 \
    --share-decoder-input-output-embed \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing .1 \
    --optimizer adam \
    --lr .001 \
    --clip-norm 1 \
    --batch-size 50 \
    --max-update 800 \
    --no-epoch-checkpoints
 ```

### Evaluation

```
fairseq-generate \
    data-bin \
    --source-lang ady.g \
    --target-lang ady.p \
    --path checkpoints/checkpoint_best.pt \
    --gen-subset test \
    --beam 8 \
    > predictions.txt
./wer.py predictions.txt
```


