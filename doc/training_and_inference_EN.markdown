# Diff-SVC(train/inference by yourself)
## 0. Setting up the environment
>Notice: The requirements files have been updated, and there are now three versions to choose from.

1. requirements.txt contains the entire environment during development and testing. It includes Torch1.12.1+cu113, and you can use pip to install it directly or remove the packages related to PyTorch inside (torch/torchvision) and then use pip to install it and use your own torch environment.
    ```
    pip install -r requirements.txt
    ```
2. **(Recommended)**: `requirements_short.txt` is a manually organized version of the one above but does not include torch itself. You can also just run the code below:
    ```
    pip install -r requirements_short.txt
    ```
3. There is a requirements list (requirements.png) compiled by @三千 under the project's root directory, which was tested on a certain brand's cloud servers. However, its torch version is NOT compatible with the latest code anymore, but the versions of the other requirements can be used as a reference.

## 1. Inference
>You can use `inference.ipynb` in the project's root directory or use `infer.py` written by @IceKyrin and adapted by the author for inference.\
Edit the parameters below in the first block:
```
config_path= 'location of config.yaml in the checkpoints archive'
# E.g.: './ckpts/nyaru/config.yaml'
# The config and checkpoints are one-to-one correspondences. Please do not use other config files.

project_name='name of the current project'
# E.g.: 'nyaru'

model_path='full path to the ckpt file'
# E.g.: './ckpts/nyaru/model_ckpt_steps_112000.ckpt'

hubert_gpu=True
# Whether or not to use GPU for HuBERT (a module in the model) during inference. It will not affect any other parts of the model.
# The current version significantly reduces the GPU usage for inferencing the HuBERT module. As full inference can be made on a 1060 6G GPU, there is no need to turn it off.
# Also, auto-slice of long audio is now supported (both inference.ipynb and infer.py support this). Audio longer than 30 seconds will be automatically sliced at silences, thanks to @IceKyrin's code.
```
### Adjustable parameters：
```
wav_fn='xxx.wav'
# The path to the input audio. The default path is in the project's root directory.

use_crepe=True
# CREPE is an F0 extraction algorithm. It has good performance but is slow. Changing this to False will use the slightly inferior but much faster Parselmouth algorithm.

thre=0.05
# CREPE's noise filtering threshold. It can be increased if the input audio is clean, but if the input audio is noisy, keep this value or decrease it. This parameter will have no effect if the previous parameter is set to False.

pndm_speedup=20
# Inference acceleration multiplier. The default number of diffusion steps is 1000, so changing this value to 10 means synthesizing in 100 steps. The default, 20, is a moderate value. This value can go up to 50x (synthesizing in 20 steps) without obvious loss in quality, but any higher may result in a significant quality loss. Note: if use_gt_mel below is enabled, make sure this value is lower than add_noise_step. This value should also be divisible by the number of diffusion steps.

key=0
# Transpose parameter. The default value is 0 (NOT 1!!). The pitch from the input audio will be shifted by {key} semitones, then synthesized. For example, to change a male voice to a female voice, this value can be set to 8 or 12, etc. (12 is to shift a whole octave up).

use_pe=True
# F0 extraction algorithm for synthesizing audio from the Mel spectrogram. Changing this to False will use the input audio's F0.
# There is a slight difference in results between using True and False. Usually, setting it to True is better, but not always. It has almost no effect on the synthesizing speed.
# (Regardless of what the value of the key parameter is, this value is always changeable and does not affect it)
# This function is not supported in 44.1kHz models and will be turned off automatically. Leaving it on will not cause any errors as well.

use_gt_mel=False
# This option is similar to the image-to-image function in AI painting. If set to True, the output audio will be a mix of the input and target speaker's voices, with the mix ratio determined by the parameter below.
# NOTE!!!: If this parameter is set to true, make sure the key parameter is set to 0 since transposing is not supported here.

add_noise_step=500
# Related to the previous parameter, it controls the ratio of the input and target voice. A value of 1 will be entirely the input voice, and a value of 1000 will be entirely the target voice. A value of around 300 will result in a roughly equal mixture of the two. (This value is not linear; if this parameter is set to a very low value, you can lower pndm_speedup for higher synthesis quality)


wav_gen='yyy.wav'
# The path to the output audio. The default is in the project's root directory. The file type can be changed by changing the file extension here.
```

If using infer.py, the way to change parameters is similar. Change values inside `__name__=='__main__'`, then run `python infer.py` in the project's root directory.
This method requires putting the input audio under raw/ and the output will under results/.

## 2. Data preparation and training
### 2.1 Data preparation
>Currently, both WAV and Ogg format audio are supported. The sampling rate is better to be higher than 24kHz. The program will automatically handle issues with sampling rates and the number of channels. The sampling rate should not be lower than 16kHz (which usually will not). \
The audio is better to be sliced into segments of 5-15 seconds. While there is no specific requirement for the audio length, it is best for them not to be too long or too short. The audio needs to be the target speaker's dry vocals without background music or other voices, preferably without excessive reverb, etc. If the audio is processed through vocal extraction, please try to keep the audio quality as high as possible. \
Currently, only single-speaker training is supported. The total audio duration should be 3 hours or above. No additional labeling is required. Just place the audio files under raw_data_dir described below. The structure of this directory does not matter; the program will locate the files by itself.

### 2.2 Editing hyperparameters
>First, make a backup copy of config.yaml (this file is for the 24kHz vocoder; use config_nsf.yaml for the 44.1kHz vocoder), then edit it: \
The parameters below might be used (using project name `nyaru` as an example):
```
K_step: 1000
# The total number of diffusion steps. Changing this is not recommended.

binary_data_dir: data/binary/nyaru
# The path to the pre-processed data: the last part needs to be changed to the current project name.

config_path: training/config.yaml
# The path to this config.yaml itself that you are using. Since data will be written into this file during the pre-processing process, this must be the full path to where the yaml file will be stored.

choose_test_manually: false
# Manually selecting a test set. It is disabled by default, and the program will automatically randomly select 5 audio files as the test set.
# If set to true, enter the prefixes of the filenames of the test files in test_prefixes. The program will use the files starting with the corresponding prefix(es) as the test set.
# This is a list and can contain multiple prefixes, e.g.
test_prefixes:
- test
- aaaa
- 5012
- speaker1024
# IMPORTANT: the test set CAN NOT be empty. To avoid unintended effects, it is recommended to avoid manually selecting the test set.

endless_ds:False
# If your dataset is too small, each epoch will pass very fast. Setting this to True will treat 1000 epochs as a single one.

hubert_path: ckpts/hubert/hubert.pt
# The path to the HuBERT model, make sure this path is correct. In most cases, the decompressed checkpoints.zip archive would put the model under the right path, so no edits are needed. The torch version is now used for inference.

hubert_gpu:True
# Whether or not to use GPU for HuBERT (a module in the model) during pre-processing. If set to False, CPU will be used, and the processing time will increase significantly. Note that whether GPU is used during inference is controlled separately in inference and not affected by this. Since HuBERT changed to the torch version, it is possible to run pre-processing and inference audio under 1 minute without exceeding VRAM limits on a 1060 6G GPU now, so it is usually not necessary to set it to False.

lr: 0.0008
# Initial learning rate: this value corresponds to a batch size of 88; if the batch size is smaller, you can lower this value a bit.

decay_steps: 20000
# For every 20,000 steps, the learning rate will decay to half the original. If the batch size is small, please increase this value.

# For a batch size of about 30-40, the recommended values are lr=0.0004，decay_steps=40000

max_frames: 42000
max_input_tokens: 6000
max_sentences: 88
max_tokens: 128000
# The batch size is calculated dynamically based on these parameters. If unsure about their exact meaning, you can change the max_sentences parameter only, which sets the maximum limit for the batch size to avoid exceeding VRAM limits.

pe_ckpt: ckpts/0102_xiaoma_pe/model_ckpt_steps_60000.ckpt
# Path to the pe model. Make sure this file exists. Refer to the inference section for its purpose.

raw_data_dir: data/raw/nyaru
# Path to the directory of the raw data before pre-processing. Please put the raw audio files under this directory. The structure inside does not matter, as the program will automatically parse it.

residual_channels: 384
residual_layers: 20
# A group of parameters that control the core network size. The higher the values, the more parameters the network has and the slower it trains, but this does not necessarily lead to better results. For larger datasets, you can change the first parameter to 512. You can experiment with them on your own. However, it is best to leave them as they are if you are not sure what you are doing.

speaker_id: nyaru
# The name of the target speaker. Currently, only single-speaker is supported. (This parameter is for reference only and has no functional impact)

use_crepe: true
# Use CREPE to extract F0 for pre-processing. Enable it for better results, or disable it for faster processing.

val_check_interval: 2000
# Inference on the test set and save checkpoints every 2000 steps.

vocoder_ckpt: ckpts/0109_hifigan_bigpopcs_hop128
# For 24kHz models, this should be the path to the directory of the corresponding vocoder. For 44.1kHz models, this should be the path to the corresponding vocoder file itself. Be careful, do not put the wrong one.

work_dir: ckpts/nyaru
# Change the last part to the project name. (Or it can also be deleted or left completely empty to generate this directory automatically, but do not put some random names)

no_fs2: true
# Simplification of the network encoder. It can reduce the model size and speed up training. No direct evidence of damage to the network performance has been found so far. Enabled by default.

```
> Do not edit the other parameters if you do not know that they do, even if you think you may know by judging from their names.

### 2.3 Data pre-processing
Run the following commands under the diff-svc directory: \
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
For pre-processing, @IceKyrin has prepared a code for processing HuBERT and other features separately. If your VRAM is insufficient to do it normally, you can run `python ./network/hubert/hubert_model.py` first and then run the pre-processing commands, which can recognize the pre-processed HuBERT features.

### 2.4 Training
#windows
```
set CUDA_VISIBLE_DEVICES=0
python run.py --config training/config.yaml --exp_name nyaru --reset
```
#linux
```
CUDA_VISIBLE_DEVICES=0 python run.py --config training/config.yaml --exp_name nyaru --reset
```
>You need to change `exp_name` to your project name and edit the config path. Please make sure that the config file used for training is the same as the one used for pre-processing.\
*Important*: After finishing training (on the cloud), if you did not pre-process the data locally, you will need to download the corresponding ckpt file AND the config file for inference. Do not use the one on your local machine since pre-processing writes data into the config file. Make sure the config file used for inference is the same as the one from pre-processing.

### 2.5 Possible issues：

>**2.5.1 'Upsample' object has no attribute 'recompute_scale_factor'**\
This issue was found in the torch version corresponding to cuda 11.3. If this issue occurs, please locate the `torch.nn.modules.upsampling.py` file in your python package (for example, in a conda environment, it is located under conda_dir\envs\environment_dir\Lib\site-packages\torch\nn\modules\upsampling.py), edit line 153-154 from
```
return F.interpolate(input, self.size, self.scale_factor, self.mode, self.align_corners,recompute_scale_factor=self.recompute_scale_factor)
```
>to
```
return F.interpolate(input, self.size, self.scale_factor, self.mode, self.align_corners)
# recompute_scale_factor=self.recompute_scale_factor)
```

>**2.5.2 no module named 'utils'**\
Please set up in your runtime environment (such as colab notebooks) as follows:
```
import os
os.environ['PYTHONPATH']='.'
!CUDA_VISIBLE_DEVICES=0 python preprocessing/binarize.py --config training/config.yaml
```
Note that this must be done in the project's root directory.

>**2.5.3 cannot load library 'libsndfile.so'**\
This is an error that may occur in a Linux environment. Please run the following command:
```
apt-get install libsndfile1 -y
```
>**2.5.4 cannot load import 'consume_prefix_in_state_dict_if_present'**\
The current torch version is too old. Please upgrade to a higher version of torch.

>**2.5.5 Data pre-processing being too slow**\
Check if `use_crepe` is enabled in config. Turning it off can significantly increase speed.\
Check if `hubert_gpu` is enabled in config.

If there are any other questions, feel free to join the QQ channel or Discord server to ask.
