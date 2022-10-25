# Diff-SVC(train/inference by yourself)
## 0.环境配置
>注意:requirements中与pytorch有关的项目(torch/torchvision/pytorch-lightning)以及onnxruntime已事先删除
```
pip install -r requirements.txt
```
>之后后请根据本地cuda自行选择对应的torch版本安装\
此项目依赖pytorch-lightning, 请记得安装\
onnxruntime有gpu版本onnxruntime-gpu, torch和cuda版本不太低可以直接
```
pip install onnxruntime-gpu
```
>其他版本请查询onnxruntime官网上的对应关系\
注意:requirements.txt中存在一些冗余依赖，暂时尚未剔除，但并没有体积十分大的，如果在意，可以参考根目录下@三千整理的一份依赖列表requirements.png，十分感谢

## 1.推理
>使用根目录下的inference.ipynb进行推理或使用@小狼整理的infer.py\
在第一个block中修改如下参数：
```
config='checkpoints压缩包中config.yaml的位置'
如'./checkpoints/nyaru/config.yaml'
config和checkpoints是一一对应的，请不要使用其他config

exp_name='这个项目的名称'
如'nyaru'

modelpath='ckpt文件的全路径'
如'./checkpoints/nyaru/model_ckpt_steps_112000.ckpt'
```
### 可调节参数：
```
wav_fn='xxx.wav'#传入音频的路径

use_crepe=True 
#crepe是一个F0算法，效果好但速度慢，改成False会使用效果稍逊于crepe但较快的parselmouth算法

thre=0.05
#crepe的噪声过滤阈值，源音频干净可适当调大，噪音多就保持这个数值或者调小，前面改成False后这个参数不起作用

hparams['pndm_speedup']=10
#推理加速算法倍数，默认是1000步，这里填成10就是只使用100步合成，是一个中规中矩的数值，这个数值可以高到50倍(20步合成)没有明显质量损失，再大可能会有可观的质量损失

key=0
#变调参数，默认为0(不是1!!)，将源音频的音高升高key个半音后合成，如男声转女生，可填入8或者12等(12就是升高一整个8度)

use_pe=True
#梅尔谱合成音频时使用的F0提取算法，如果改成False将使用源音频的F0
这里填True和False合成会略有差异，通常是True会好些，但也不尽然，对合成速度几乎无影响
(无论key填什么 这里都是可以自由选择的，不影响)

use_gt_mel=False
#这个选项类似于AI画图的图生图功能，如果打开，产生的音频将是输入声音与目标说话人声音的混合，混合比例由下一个参数确定
注意!!!：这个参数如果改成True，请确保key填成0，不支持变调

add_noise_step=500
#与上个参数有关，控制两种声音的比例，填入1是完全的源声线，填入1000是完全的目标声线，能听出来是两者均等混合的数值大约在300附近(并不是线性的，另外这个参数如果调的很小，可以把pndm加速倍率调低，增加合成质量)

wav_gen='yyy.wav'#输出音频的路径
```

## 2.数据预处理与训练
### 2.1
>首先请备份一份config.yaml，然后修改它：\
可能会用到的参数如下(以工程名为nyaru为例):
```
K_step: 1000
#diffusion过程总的step,建议不要修改

binary_data_dir: data/binary/nyaru
预处理后数据的存放地址:需要将后缀改成工程名字

config_path: training/config.yaml
你要使用的这份yaml自身的地址，由于预处理过程中会写入数据，所以这个地址务必修改成将要存放这份yaml文件的完整路径

choose_test_manually: false
手动选择测试集，默认关闭，自动随机抽取5条音频作为测试集。
如果改为ture，请在test_prefixes:中填入测试数据的文件名前缀，程序会将以对应前缀开头的文件作为测试集
这是个列表，可以填多个前缀，如：
test_prefixes:
- test
- aaaa
- 5012
- speaker1024
重要：测试集*不可以*为空，为了不产生意外影响，建议尽量不要手动选择测试集

hubert_path: checkpoints/hubert/hubert.onnx
hubert模型的存放地址，确保这个路径是对的，一般解压checkpoints包之后就是这个路径不需要改

lr: 0.0008
#初始的学习率:这个数字对应于88的batchsize，如果batchsize更小，可以调低这个数值一些

decay_steps: 20000
每20000步学习率衰减为原来的一半，如果batchsize比较小，请调大这个数值

max_frames: 42000
max_input_tokens: 6000
max_sentences: 88
max_tokens: 128000
#batchsize是由这几个参数动态算出来的，如果不太清楚具体含义，可以只改动max_sentences这个参数，填入batchsize的最大限制值，以免炸显存

pe_ckpt: checkpoints/0102_xiaoma_pe/model_ckpt_steps_60000.ckpt
#pe模型路径，确保这个文件存在，具体作用参考inference部分

raw_data_dir: data/raw/nyaru
#存放预处理前原始数据的位置，请将原始wav数据放在这个目录下，内部文件结构无所谓，会自动解构

speaker_id: nyaru
训练的说话人名字，目前只支持单说话人，请在这里填写

use_crepe: true
#在数据预处理中使用crepe提取F0,追求效果请打开，追求速度可以关闭

val_check_interval: 2000
每2000steps推理测试集并保存ckpt

work_dir: checkpoints/nyaru
#修改后缀为工程名
```
>其他的参数如果你不知道它是做什么的，请不要修改

### 2.2 数据预处理
在diff-svc的目录下执行以下命令：\
#windows
```
set PYTHONPATH=.
set CUDA_VISIBLE_DEVICES=0 
python preprocessing/binarize.py --config training/config.yaml
```
#linux
```
export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=0 python preprocessing/binarize.py --config training/config.yaml
```
>注意：如果此条命令报找不到utils的错误，请在如colab笔记本的环境中以如下方式设置:
```
import os
os.environ['PYTHONPATH']='.'
!CUDA_VISIBLE_DEVICES=0 python preprocessing/binarize.py --config training/config.yaml
```
### 2.3 训练
#windows
```
set CUDA_VISIBLE_DEVICES=0 
python run.py --config training/train.yaml --exp_name nyaru --reset  
```
#linux
```
CUDA_VISIBLE_DEVICES=0 python run.py --config training/config.yaml --exp_name nyaru --reset 
```
>需要将exp_name改为你的工程名，并修改config路径，请确保和预处理使用的是同一个config文件\
*重要* ：若不在本地训练，训练完成后，除了需要下载对应的ckpt文件，也需要将此config下载下来，作为推理时使用的config

如有其他问题，请扫描github仓库界面下方的二维码询问。
