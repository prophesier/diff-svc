# Diff-SVC
Singing Voice Conversion via diffusion model
## updates:
>2022.11.1 优化推理部分\
2022.10.29 整理推理部分，添加长音频自动切片功能。\
2022.10.28 将hubert的onnx推理迁移为torch推理，并整理推理逻辑。\
<font color=#FFA500>如原先下载过onnx的hubert模型需重新下载并替换为pt模型</font>，config不需要改，目前可以实现1060 6G显存的直接GPU推理与预处理，详情请查看文档。\
2022.10.27 更新依赖文件，去除冗余依赖。\
2022.10.27 修复了一个严重错误，曾导致在gpu服务器上hubert仍使用cpu推理，速度减慢3-5倍，影响预处理与推理，不影响训练\
2022.10.26 修复windows上预处理数据在linux上无法使用的问题，更新部分文档\
2022.10.25 编写推理/训练详细文档，修改整合部分代码，增加对ogg格式音频的支持(无需与wav区分，直接使用即可)\
2022.10.24 支持对自定义数据集的训练，并精简代码\
2022.10.22 完成对opencpop数据集的训练并创建仓库

## 推理：

>查看./inference.ipynb


## 预处理:
```
export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=0 python preprocessing/binarize.py --config training/config.yaml
```
## 训练:
```
CUDA_VISIBLE_DEVICES=0 python run.py --config training/config.yaml --exp_name [your project name] --reset 
```
详细训练过程和各种参数介绍请查看[推理与训练说明](./doc/train_and_inference.markdown)
### 已训练模型
>目前po主已在opencpop数据集和猫雷直播数据集进行过训练，并有多人进行过其他数据集训练和测试。部分ckpt文件、demo音频和推理训练所需的其他文件请在下方QQ频道内下载\
使用QQ扫描此二维码(如不能加入，请尝试一个合适的网络环境):
<img src="./ckpt.png" width=256/>

## Acknowledgements
>项目基于[diffsinger原仓库](https://github.com/MoonInTheRiver/DiffSinger)、[diffsinger(openvpi维护版)](https://github.com/openvpi/DiffSinger)、[soft-vc](https://github.com/bshall/soft-vc)开发.\
同时也十分感谢openvpi成员在开发训练过程中给予的帮助。
