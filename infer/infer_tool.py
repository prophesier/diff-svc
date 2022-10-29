import logging
import os
import shutil
import subprocess
import time

import numpy as np
import torch
import torchaudio

import utils
from modules.fastspeech.pe import PitchExtractor
from network.diff.candidate_decoder import FFT
from network.diff.diffusion import GaussianDiffusion
from network.diff.net import DiffNet
from network.vocoders.base_vocoder import VOCODERS
from network.vocoders.base_vocoder import get_vocoder_cls
from preprocessing.data_gen_utils import get_pitch_parselmouth, get_pitch_crepe
from preprocessing.hubertinfer import Hubertencoder
from utils.hparams import hparams
from utils.hparams import set_hparams
from utils.pitch_utils import denorm_f0
from utils.pitch_utils import norm_interp_f0

logging.getLogger('matplotlib').setLevel(logging.WARNING)


def timeit(func):
    def run(*args, **kwargs):
        t = time.time()
        res = func(*args, **kwargs)
        print('executing \'%s\' costed %.3fs' % (func.__name__, time.time() - t))
        return res

    return run


def cut_wav(raw_audio_path, out_audio_name, input_wav_path, cut_time):
    raw_audio, raw_sr = torchaudio.load(raw_audio_path)
    if raw_audio.shape[-1] / raw_sr > cut_time:
        subprocess.Popen(
            f"python ./infer/slicer.py {raw_audio_path} --out_name {out_audio_name} --out {input_wav_path}  --db_thresh -30",
            shell=True).wait()
    else:
        shutil.copy(raw_audio_path, f"{input_wav_path}/{out_audio_name}-00.wav")


def get_end_file(dir_path, end):
    file_lists = []
    for root, dirs, files in os.walk(dir_path):
        files = [f for f in files if f[0] != '.']
        dirs[:] = [d for d in dirs if d[0] != '.']
        for f_file in files:
            if f_file.endswith(end):
                file_lists.append(os.path.join(root, f_file).replace("\\", "/"))
    return file_lists


def del_temp_wav(path_data):
    for i in get_end_file(path_data, "wav"):  # os.listdir(path_data)#返回一个列表，里面是当前目录下面的所有东西的相对路径
        os.remove(i)


def fill_a_to_b(a, b):
    if len(a) < len(b):
        for _ in range(0, len(b) - len(a)):
            a.append(a[0])


def mkdir(paths: list):
    for path in paths:
        if not os.path.exists(path):
            os.mkdir(path)


class Svc:
    def __init__(self, project_name, model_path):
        self.DIFF_DECODERS = {
            'wavenet': lambda hp: DiffNet(hp['audio_num_mel_bins']),
            'fft': lambda hp: FFT(
                hp['hidden_size'], hp['dec_layers'], hp['dec_ffn_kernel_size'], hp['num_heads']),
        }

        self.model_path = model_path
        self.dev = torch.device("cuda")

        self._ = set_hparams(config=f'checkpoints/{project_name}/config.yaml', exp_name=project_name, infer=True,
                             reset=True,
                             hparams_str='',
                             print_hparams=False)

        self.mel_bins = hparams['audio_num_mel_bins']
        self.model = GaussianDiffusion(
            phone_encoder=Hubertencoder(hparams['hubert_path']),
            out_dims=self.mel_bins, denoise_fn=self.DIFF_DECODERS[hparams['diff_decoder_type']](hparams),
            timesteps=hparams['timesteps'],
            K_step=hparams['K_step'],
            loss_type=hparams['diff_loss_type'],
            spec_min=hparams['spec_min'], spec_max=hparams['spec_max'],
        )
        self.load_ckpt()
        self.model.cuda()
        self.hubert = Hubertencoder(hparams['hubert_path'])
        self.pe = PitchExtractor().cuda()
        utils.load_ckpt(self.pe, hparams['pe_ckpt'], 'model', strict=True)
        self.pe.eval()
        self.vocoder = get_vocoder_cls(hparams)()

    def load_ckpt(self, model_name='model', force=True, strict=True):
        utils.load_ckpt(self.model, self.model_path, model_name, force, strict)

    @timeit
    def infer(self, in_path, key, acc, use_pe=True, **kwargs):
        batch = self.pre(in_path, acc)
        spk_embed = batch.get('spk_embed') if not hparams['use_spk_id'] else batch.get('spk_ids')
        hubert = batch['hubert']
        ref_mels = batch["mels"]
        mel2ph = batch['mel2ph']
        batch['f0'] = batch['f0'] + (key / 12)
        f0 = batch['f0']
        uv = batch['uv']
        outputs = self.model(
            hubert.cuda(), spk_embed=spk_embed, mel2ph=mel2ph.cuda(), f0=f0.cuda(), uv=uv.cuda(),
            ref_mels=ref_mels.cuda(),
            infer=True, **kwargs)
        batch['outputs'] = self.model.out2mel(outputs['mel_out'])
        batch['mel2ph_pred'] = outputs['mel2ph']
        batch['f0_gt'] = denorm_f0(batch['f0'], batch['uv'], hparams)
        if use_pe:

            batch['f0_pred'] = self.pe(outputs['mel_out'])['f0_denorm_pred'].detach()
        else:
            batch['f0_pred'] = outputs.get('f0_denorm')
        return self.after_infer(batch)

    @timeit
    def after_infer(self, prediction):
        for k, v in prediction.items():
            if type(v) is torch.Tensor:
                prediction[k] = v.cpu().numpy()

        # remove paddings
        mel_pred = prediction["outputs"]
        mel_pred_mask = np.abs(mel_pred).sum(-1) > 0
        mel_pred = mel_pred[mel_pred_mask]
        mel_pred = np.clip(mel_pred, hparams['mel_vmin'], hparams['mel_vmax'])

        f0_pred = prediction.get("f0_pred")
        if len(f0_pred) > len(mel_pred_mask):
            f0_pred = f0_pred[:len(mel_pred_mask)]
        f0_pred = f0_pred[mel_pred_mask]
        torch.cuda.is_available() and torch.cuda.empty_cache()
        wav_pred = self.vocoder.spec2wav(mel_pred, f0=f0_pred)
        return wav_pred

    def temporary_dict2processed_input(self, item_name, temp_dict, use_crepe=True, thre=0.05):
        '''
            process data in temporary_dicts
        '''

        binarization_args = hparams['binarization_args']

        def get_pitch(wav, mel):
            # get ground truth f0 by self.get_pitch_algorithm
            if use_crepe:
                torch.cuda.is_available() and torch.cuda.empty_cache()
                gt_f0, coarse_f0 = get_pitch_crepe(wav, mel, hparams, thre)  #
            else:
                gt_f0, coarse_f0 = get_pitch_parselmouth(wav, mel, hparams)
            processed_input['f0'] = gt_f0
            processed_input['pitch'] = coarse_f0

        def get_align(mel, phone_encoded):
            mel2ph = np.zeros([mel.shape[0]], int)
            start_frame = 1
            ph_durs = mel.shape[0] / phone_encoded.shape[0]
            if hparams['debug']:
                print(mel.shape, phone_encoded.shape, mel.shape[0] / phone_encoded.shape[0])
            for i_ph in range(phone_encoded.shape[0]):
                end_frame = int(i_ph * ph_durs + ph_durs + 0.5)
                mel2ph[start_frame:end_frame + 1] = i_ph + 1
                start_frame = end_frame + 1

            processed_input['mel2ph'] = mel2ph

        if hparams['vocoder'] in VOCODERS:
            wav, mel = VOCODERS[hparams['vocoder']].wav2spec(temp_dict['wav_fn'])
        else:
            wav, mel = VOCODERS[hparams['vocoder'].split('.')[-1]].wav2spec(temp_dict['wav_fn'])
        processed_input = {
            'item_name': item_name, 'mel': mel,
            'sec': len(wav) / hparams['audio_sample_rate'], 'len': mel.shape[0]
        }
        processed_input = {**temp_dict, **processed_input}  # merge two dicts

        if binarization_args['with_f0']:
            get_pitch(wav, mel)
        if binarization_args['with_hubert']:
            st = time.time()
            hubert_encoded = processed_input['hubert'] = self.hubert.encode(temp_dict['wav_fn'])
            et = time.time()
            dev = 'cuda' if hparams['hubert_gpu'] and torch.cuda.is_available() else 'cpu'
            print(f'hubert (on {dev}) time used {et - st}')

            if binarization_args['with_align']:
                get_align(mel, hubert_encoded)
        return processed_input

    def pre(self, in_path, accelerate):
        temp_dict = self.temporary_dict2processed_input(*file2temporary_dict(in_path), use_crepe=True, thre=0.05)
        hparams['pndm_speedup'] = accelerate
        batch = processed_input2batch([getitem(temp_dict)])
        return batch


def file2temporary_dict(wav_fn):
    '''
        read from file, store data in temporary dicts
    '''
    song_info = wav_fn.split('/')
    item_name = raw_item_name = song_info[-1].split('.')[-2]
    temp_dict = {}

    temp_dict['wav_fn'] = wav_fn
    temp_dict['spk_id'] = 'opencpop'

    return item_name, temp_dict


def getitem(item):
    max_frames = hparams['max_frames']
    spec = torch.Tensor(item['mel'])[:max_frames]
    energy = (spec.exp() ** 2).sum(-1).sqrt()
    mel2ph = torch.LongTensor(item['mel2ph'])[:max_frames] if 'mel2ph' in item else None
    f0, uv = norm_interp_f0(item["f0"][:max_frames], hparams)
    hubert = torch.Tensor(item['hubert'][:hparams['max_input_tokens']])
    pitch = torch.LongTensor(item.get("pitch"))[:max_frames]
    sample = {
        "item_name": item['item_name'],
        "hubert": hubert,
        "mel": spec,
        "pitch": pitch,
        "energy": energy,
        "f0": f0,
        "uv": uv,
        "mel2ph": mel2ph,
        "mel_nonpadding": spec.abs().sum(-1) > 0,
    }
    return sample


def processed_input2batch(samples):
    '''
        Args:
            samples: one batch of processed_input
        NOTE:
            the batch size is controlled by hparams['max_sentences']
    '''
    if len(samples) == 0:
        return {}
    item_names = [s['item_name'] for s in samples]
    hubert = utils.collate_2d([s['hubert'] for s in samples], 0.0)
    f0 = utils.collate_1d([s['f0'] for s in samples], 0.0)
    pitch = utils.collate_1d([s['pitch'] for s in samples])
    uv = utils.collate_1d([s['uv'] for s in samples])
    energy = utils.collate_1d([s['energy'] for s in samples], 0.0)
    mel2ph = utils.collate_1d([s['mel2ph'] for s in samples], 0.0) \
        if samples[0]['mel2ph'] is not None else None
    mels = utils.collate_2d([s['mel'] for s in samples], 0.0)
    mel_lengths = torch.LongTensor([s['mel'].shape[0] for s in samples])

    batch = {
        'item_name': item_names,
        'nsamples': len(samples),
        'hubert': hubert,
        'mels': mels,
        'mel_lengths': mel_lengths,
        'mel2ph': mel2ph,
        'energy': energy,
        'pitch': pitch,
        'f0': f0,
        'uv': uv,
    }
    return batch
