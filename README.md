# Mean Prototypical Networks for Text Classification
code for paper submit to CCL 2019

## data preprocess
`# python preprocess.py --dir data/ag_news_csv --one-based-class`

## train model
```bash
# CUDA_VISIBLE_DEVICES=0 python train.py \
    --dir data/yelp_review_full_csv \
    --class-num 5  \
    --cuda \ 
    --log-freq 100 \
    --encoder gru \
    --rnn-dim 100 \
    --embed-dim 100 \
    --embed-drop 0.5 \
    --rnn-dropout 0.1 \
    --epochs 50 \ 
    --proto-decay 0.99 \
    --samples-per-class 64 \
    --support-num 32 \
    --iters-per-epoch 2500 \
    --proto-dim 50 \
    --ckpt ./ckpt-yelp-f-p50 
```

## predict
```bash
# CUDA_VISIBLE_DEVICES=0 python train.py \
    --dir data/yelp_review_full_csv \
    --ckpt ./ckpt-yelp-f-p50/best.ckpt
    --cuda

```