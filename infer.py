import librosa
import soundfile as sf
import torch

import utils
from infer_tool import infer_tool
from modules.fastspeech.pe import PitchExtractor
from network.diff.candidate_decoder import FFT
from network.diff.diffusion import GaussianDiffusion
from network.diff.net import DiffNet
from preprocessing.hubertinfer import Hubertencoder
from utils.hparams import hparams, set_hparams
from utils.pitch_utils import denorm_f0
from network.vocoders.base_vocoder import get_vocoder_cls, BaseVocoder


DIFF_DECODERS = {
    'wavenet': lambda hp: DiffNet(hp['audio_num_mel_bins']),
    'fft': lambda hp: FFT(
        hp['hidden_size'], hp['dec_layers'], hp['dec_ffn_kernel_size'], hp['num_heads']),
}

# 工程文件夹名，训练时用的那个
project_name = "SVC"
# 改一下steps就行
model_path = f'./checkpoints/{project_name}/model_ckpt_steps_240000.ckpt'

# 输入输出文件名
wav_fn = '祈.wav'
wav_gen = f'out_{wav_fn}'

# 加速倍数
accelerate = 50

# 下面不动
_ = set_hparams(config=f'checkpoints/{project_name}/config.yaml', exp_name=project_name, infer=True, reset=True,
                hparams_str='',
                print_hparams=False)

mel_bins = hparams['audio_num_mel_bins']
encoder=Hubertencoder(hparams['hubert_path'])
model = GaussianDiffusion(
    phone_encoder=encoder,
    out_dims=mel_bins, denoise_fn=DIFF_DECODERS[hparams['diff_decoder_type']](hparams),
    timesteps=hparams['timesteps'],
    K_step=hparams['K_step'],
    loss_type=hparams['diff_loss_type'],
    spec_min=hparams['spec_min'], spec_max=hparams['spec_max'],
)

dev = "cuda:0" if torch.cuda.is_available() else "cpu"
utils.load_ckpt(model,model_path,'model',force=True, strict=True)
model.to(dev)
pe = PitchExtractor().to(dev)
utils.load_ckpt(pe, hparams['pe_ckpt'], 'model', strict=True)
pe.eval()

vocoder: BaseVocoder = get_vocoder_cls(hparams)()
inf=infer_tool(encoder,model,pe,vocoder)
print('model loaded')

demo_audio, sr = librosa.load(wav_fn, sr=None)
temp_dict = tempdict=inf.make_dict(wav_fn,use_crepe=True,thre=0.05)

hparams['pndm_speedup'] = accelerate
f0_tst,f0_pred,audio=inf.infer(tempdict,key=0,use_pe=True,use_gt_mel=False,add_noise_step=1000)

sf.write(wav_gen, audio, 24000, 'PCM_16')
