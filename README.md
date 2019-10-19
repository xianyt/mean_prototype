# Mean Prototypical Networks for Text Classification
code for paper submit to CCL 2019

## data preprocess
`# python preprocess.py --dir data/ag_news_csv --one-based-class`

## train model
```bash
# CUDA_VISIBLE_DEVICES=0 python train.py \
    --dir data/ag_news_csv \
    --class-num 4  \
    --cuda \ 
    --log-freq 10 \
    --encoder gru \
    --rnn-dim 384 \
    --embed-dim 100 \
    --embed-drop 0.5 \
    --rnn-dropout 0.1 \
    --epochs 50 \ 
    --proto-decay 0.99 \
    --samples-per-class 64 \
    --support-num 32 \
    --iters-per-epoch 2500 \
    --proto-dim 50 \
    --ckpt ./ag
```

## predict
```bash
# CUDA_VISIBLE_DEVICES=0 python train.py \
    --dir data/ag_news_csv \
    --ckpt ./ckpt/best.ckpt
    --cuda

```
