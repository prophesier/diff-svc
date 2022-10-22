# Diff-SVC
Singing Voice Conversion via diffusion model

## 推理：

>查看./inference.ipynb


## 预处理:

>export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=0 python preprocessing/binarize.py --config preprocessing/preprocess.yaml

## 训练:

>CUDA_VISIBLE_DEVICES=0 python run.py --config training/train.yaml --exp_name SVC --reset 

