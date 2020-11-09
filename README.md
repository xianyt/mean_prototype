# Mean Prototypical Networks for Text Classification
code for paper “Mean Prototypical Networks for Text Classification”

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

## Cite

If you use the code, please cite this paper:

线岩团,相艳,余正涛,文永华,王红斌,张亚飞. 用于文本分类的均值原型网络[J]. 中文信息学报, 2020, 34(6): 73-80,88. 

XIAN Yantuan, XIANG Yan, YU Zhengtao, WEN Yonghua, WANG Hongbin, ZHANG Yafei. Mean Prototypical Networks for Text Classification. , 2020, 34(6): 73-80,88.
