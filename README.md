# Diff-SVC
Singing Voice Conversion via diffusion model

## updates:
>2022.12.4 44.1kHz声码器开放申请，正式提供对44.1kHz的支持\
2022.11.28 增加了默认打开的no_fs2选项，可优化部分网络，提升训练速度、缩减模型体积，对于未来新训练的模型有效\
2022.11.23 修复了一个重大bug，曾导致可能将用于推理的原始gt音频转变采样率为22.05kHz,对于由此造成的影响我们表示十分抱歉，请务必检查自己的测试音频，并使用更新后的代码\
2022.11.22 修复了很多bug，其中有几个影响推理效果重大的bug\
2022.11.20 增加对推理时多数格式的输入和保存，无需手动借助其他软件转换\
2022.11.13 修正中断后读取模型的epoch/steps显示问题，添加f0处理的磁盘缓存，添加实时变声推理的支持文件\
2022.11.11 修正切片时长误差，补充对44.1khz的适配, 增加对contentvec的支持\
2022.11.4 添加梅尔谱保存功能\
2022.11.2 整合新声码器代码，更新parselmouth算法\
2022.10.29 整理推理部分，添加长音频自动切片功能。\
2022.10.28 将hubert的onnx推理迁移为torch推理，并整理推理逻辑。\
<font color=#FFA500>如原先下载过onnx的hubert模型需重新下载并替换为pt模型</font>，config不需要改，目前可以实现1060 6G显存的直接GPU推理与预处理，详情请查看文档。\
2022.10.27 更新依赖文件，去除冗余依赖。\
2022.10.27 修复了一个严重错误，曾导致在gpu服务器上hubert仍使用cpu推理，速度减慢3-5倍，影响预处理与推理，不影响训练\
2022.10.26 修复windows上预处理数据在linux上无法使用的问题，更新部分文档\
2022.10.25 编写推理/训练详细文档，修改整合部分代码，增加对ogg格式音频的支持(无需与wav区分，直接使用即可)\
2022.10.24 支持对自定义数据集的训练，并精简代码\
2022.10.22 完成对opencpop数据集的训练并创建仓库

## 注意事项/Notes：
>本项目是基于学术交流目的建立，并非为生产环境准备，不对由此项目模型产生的任何声音的版权问题负责。\
如将本仓库代码二次分发，或将由此项目产出的任何结果公开发表(包括但不限于视频网站投稿)，请注明原作者及代码来源(此仓库)。\
如果将此项目用于任何其他企划，请提前联系并告知本仓库作者,十分感谢。\
>This project is established for academic exchange purposes and is not intended for production environments. We are not responsible for any copyright issues arising from the sound produced by this project's model. \
If you redistribute the code in this repository or publicly publish any results produced by this project (including but not limited to video website submissions), please indicate the original author and source code (this repository). \
If you use this project for any other plans, please contact and inform the author of this repository in advance. Thank you very much.

## 推理/inference：

>查看./inference.ipynb


## 预处理/preprocessing:
```
export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=0 python preprocessing/binarize.py --config training/config.yaml
```
## 训练/training:
```
CUDA_VISIBLE_DEVICES=0 python run.py --config training/config.yaml --exp_name [your project name] --reset 
```
详细训练过程和各种参数介绍请查看[推理与训练说明](./doc/train_and_inference.markdown)\
Please refer to the [Inference and Training Instructions](./doc/training_and_inference_EN.markdown) for a detailed training process and introduction to various parameters.Thank you for the translation provided by @ρoem.
### 已训练模型/trained models
>目前本项目已在众多数据集进行过训练和测试。部分ckpt文件、demo音频和推理训练所需的其他文件请在下方QQ频道内下载\
使用QQ扫描此二维码(如不能加入，请尝试一个合适的网络环境):
This project has been trained and tested on many datasets. You can download the ckpt files, demo audio, and other files required for inference and training in the QQ channel below by using QQ to scan this QR code (if you cannot join, please try a suitable network environment).\
<img src="./ckpt.jpg" width=256/>\
For English support, you can join this discord: 

[![Discord](https://img.shields.io/discord/1044927142900809739?color=%23738ADB&label=Discord&style=for-the-badge)](https://discord.gg/jvA5c2xzSE)

## Acknowledgements
>项目基于[diffsinger](https://github.com/MoonInTheRiver/DiffSinger)、[diffsinger(openvpi维护版)](https://github.com/openvpi/DiffSinger)、[soft-vc](https://github.com/bshall/soft-vc)开发.\
同时也十分感谢openvpi成员在开发训练过程中给予的帮助。\
This project is based on [diffsinger](https://github.com/MoonInTheRiver/DiffSinger), [diffsinger (openvpi maintenance version)](https://github.com/openvpi/DiffSinger), and [soft-vc](https://github.com/bshall/soft-vc). We would also like to thank the openvpi members for their help during the development and training process. 
>注意：此项目与同名论文[DiffSVC](https://arxiv.org/abs/2105.13871)无任何联系，请勿混淆！\
Note: This project has no connection with the paper of the same name [DiffSVC](https://arxiv.org/abs/2105.13871), please do not confuse them!