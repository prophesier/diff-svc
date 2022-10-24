# Diff-SVC
Singing Voice Conversion via diffusion model

## 推理：

>查看./inference.ipynb


## 预处理:

>export PYTHONPATH=.\
CUDA_VISIBLE_DEVICES=0 python preprocessing/binarize.py --config preprocessing/preprocess.yaml

## 训练:

>CUDA_VISIBLE_DEVICES=0 python run.py --config training/train.yaml --exp_name SVC --reset 

### 已训练模型
>checkpoints trained on opencpop dataset can be found here(QQ channel)\
please scan the QRcode with QQ:\
<img src="./ckpt.png" width=256/>

## Acknowledgements
>项目基于[diffsinger原仓库](https://github.com/MoonInTheRiver/DiffSinger)、[diffsinger(openvpi维护版)](https://github.com/openvpi/DiffSinger)开发.\
十分感谢openvpi成员在开发训练过程中给予的帮助。