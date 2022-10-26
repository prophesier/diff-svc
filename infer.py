import librosa
import soundfile as sf
import torch

import utils
from infer_tool import *
from modules.fastspeech.pe import PitchExtractor
from network.diff.candidate_decoder import FFT
from network.diff.diffusion import GaussianDiffusion
from network.diff.net import DiffNet
from preprocessing.hubertinfer import Hubertencoder
from utils.hparams import hparams, set_hparams
from utils.pitch_utils import denorm_f0


def test_step(sample, key, use_pe=True, **kwargs):
    spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_ids')
    hubert = sample['hubert']
    mel2ph, uv, f0 = None, None, None
    ref_mels = sample["mels"]
    mel2ph = sample['mel2ph']
    sample['f0'] = sample['f0'] + (key / 12)
    f0 = sample['f0']
    uv = sample['uv']
    outputs = model(
        hubert.cuda(), spk_embed=spk_embed, mel2ph=mel2ph.cuda(), f0=f0.cuda(), uv=uv.cuda(), ref_mels=ref_mels.cuda(),
        infer=True, **kwargs)
    sample['outputs'] = model.out2mel(outputs['mel_out'])
    sample['mel2ph_pred'] = outputs['mel2ph']
    sample['f0_gt'] = denorm_f0(sample['f0'], sample['uv'], hparams)
    if use_pe:
        sample['f0_pred'] = pe(outputs['mel_out'])[
            'f0_denorm_pred'].detach()  # pe(ref_mels.cuda())['f0_denorm_pred'].detach()#
    else:
        sample['f0_pred'] = outputs.get('f0_denorm')
    return after_infer(sample)


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
hparams['hubert_gpu']=False
mel_bins = hparams['audio_num_mel_bins']
model = GaussianDiffusion(
    phone_encoder=Hubertencoder(hparams['hubert_path']),
    out_dims=mel_bins, denoise_fn=DIFF_DECODERS[hparams['diff_decoder_type']](hparams),
    timesteps=hparams['timesteps'],
    K_step=hparams['K_step'],
    loss_type=hparams['diff_loss_type'],
    spec_min=hparams['spec_min'], spec_max=hparams['spec_max'],
)

dev = "cuda:0" if torch.cuda.is_available() else "cpu"
load_ckpt(model, model_path, strict=True)
model.to(dev)
pe = PitchExtractor().to(dev)
utils.load_ckpt(pe, hparams['pe_ckpt'], 'model', strict=True)
pe.eval()
print('model loaded')

demo_audio, sr = librosa.load(wav_fn, sr=None)
temp_dict = temporary_dict2processed_input(*file2temporary_dict(wav_fn), use_crepe=True, thre=0.05)

hparams['pndm_speedup'] = accelerate
batch = processed_input2batch([getitem(temp_dict)])
f0_tst, f0_pred, audio = test_step(batch, key=0, use_pe=True, use_gt_mel=False, add_noise_step=500)

sf.write(wav_gen, audio, 24000, 'PCM_16')
