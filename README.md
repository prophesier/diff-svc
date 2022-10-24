# Diff-SVC
Singing Voice Conversion via diffusion model
## update:
>现已支持对自定义数据集的训练\
修改config中的配置，预处理时将自动将统计数据写入config中，若需要移动config位置，请修改config中对自身位置的引用\
wav数据直接放在raw_data_dir下即可，自适应任何目录结构
其他配置项如有需要请参考config中的名称修改
## 推理：

>查看./inference.ipynb


## 预处理:

>export PYTHONPATH=.\
CUDA_VISIBLE_DEVICES=0 python preprocessing/binarize.py --config training/config.yaml

## 训练:

>CUDA_VISIBLE_DEVICES=0 python run.py --config training/config.yaml --exp_name [your project name] --reset 

### 已训练模型
>checkpoints trained on opencpop dataset(and others in future) can be found here(QQ channel)\
please scan the QRcode with QQ:
<img src="./ckpt.png" width=256/>

## Acknowledgements
>项目基于[diffsinger原仓库](https://github.com/MoonInTheRiver/DiffSinger)、[diffsinger(openvpi维护版)](https://github.com/openvpi/DiffSinger)开发.\
十分感谢openvpi成员在开发训练过程中给予的帮助。