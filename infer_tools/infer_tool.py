import hashlib
import json
import os
import time
from io import BytesIO

import librosa
import numpy as np
import soundfile
import torch

import utils
from pathlib import Path
from modules.fastspeech.pe import PitchExtractor
from network.diff.candidate_decoder import FFT
from network.diff.diffusion import GaussianDiffusion
from network.diff.net import DiffNet
from network.vocoders.base_vocoder import VOCODERS, get_vocoder_cls
from preprocessing.data_gen_utils import get_pitch_parselmouth, get_pitch_crepe
from preprocessing.hubertinfer import Hubertencoder
from utils.hparams import hparams, set_hparams
from utils.pitch_utils import denorm_f0, norm_interp_f0


def read_temp(file_name):
    if not os.path.exists(file_name):
        with open(file_name, "w") as f:
            f.write(json.dumps({"info": "temp_dict"}))
        return {}
    else:
        with open(file_name, "r") as f:
            data = f.read()
        data_dict = json.loads(data)
        if os.path.getsize(file_name) > 50 * 1024 * 1024:
            f_name = file_name.split("/")[-1]
            print(f"clean {f_name}")
            for wav_hash in list(data_dict.keys()):
                if int(time.time()) - int(data_dict[wav_hash]["time"]) > 14 * 24 * 3600:
                    del data_dict[wav_hash]
        return data_dict


f0_dict = read_temp("./infer_tools/f0_temp.json")


def write_temp(file_name, data):
    with open(file_name, "w") as f:
        f.write(json.dumps(data))


def timeit(func):
    def run(*args, **kwargs):
        t = time.time()
        res = func(*args, **kwargs)
        print('executing \'%s\' costed %.3fs' % (func.__name__, time.time() - t))
        return res

    return run


def format_wav(audio_path):
    if Path(audio_path).suffix=='.wav':
        return
    raw_audio, raw_sample_rate = librosa.load(audio_path, mono=True,sr=None)
    soundfile.write(Path(audio_path).with_suffix(".wav"), raw_audio, raw_sample_rate)


def fill_a_to_b(a, b):
    if len(a) < len(b):
        for _ in range(0, len(b) - len(a)):
            a.append(a[0])


def get_end_file(dir_path, end):
    file_lists = []
    for root, dirs, files in os.walk(dir_path):
        files = [f for f in files if f[0] != '.']
        dirs[:] = [d for d in dirs if d[0] != '.']
        for f_file in files:
            if f_file.endswith(end):
                file_lists.append(os.path.join(root, f_file).replace("\\", "/"))
    return file_lists


def mkdir(paths: list):
    for path in paths:
        if not os.path.exists(path):
            os.mkdir(path)


def get_md5(content):
    return hashlib.new("md5", content).hexdigest()


class Svc:
    def __init__(self, project_name, config_name, hubert_gpu, model_path):
        self.project_name = project_name
        self.DIFF_DECODERS = {
            'wavenet': lambda hp: DiffNet(hp['audio_num_mel_bins']),
            'fft': lambda hp: FFT(
                hp['hidden_size'], hp['dec_layers'], hp['dec_ffn_kernel_size'], hp['num_heads']),
        }

        self.model_path = model_path
        self.dev = torch.device("cuda")

        self._ = set_hparams(config=config_name, exp_name=self.project_name, infer=True,
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
        hparams['hubert_gpu'] = hubert_gpu
        self.hubert = Hubertencoder(hparams['hubert_path'])
        self.pe = PitchExtractor().cuda()
        utils.load_ckpt(self.pe, hparams['pe_ckpt'], 'model', strict=True)
        self.pe.eval()
        self.vocoder = get_vocoder_cls(hparams)()

    def load_ckpt(self, model_name='model', force=True, strict=True):
        utils.load_ckpt(self.model, self.model_path, model_name, force, strict)

    def infer(self, in_path, key, acc, use_pe=True, use_crepe=True, thre=0.05, singer=False, **kwargs):
        batch = self.pre(in_path, acc, use_crepe, thre)
        spk_embed = batch.get('spk_embed') if not hparams['use_spk_id'] else batch.get('spk_ids')
        hubert = batch['hubert']
        ref_mels = batch["mels"]
        mel2ph = batch['mel2ph']
        batch['f0'] = batch['f0'] + (key / 12)
        batch['f0'][batch['f0']>np.log2(hparams['f0_max'])]=0
        f0 = batch['f0']
        uv = batch['uv']
        @timeit
        def diff_infer():
            outputs = self.model(
                hubert.cuda(), spk_embed=spk_embed, mel2ph=mel2ph.cuda(), f0=f0.cuda(), uv=uv.cuda(),
                ref_mels=ref_mels.cuda(),
                infer=True, **kwargs)
            return outputs
        outputs=diff_infer()
        batch['outputs'] = self.model.out2mel(outputs['mel_out'])
        batch['mel2ph_pred'] = outputs['mel2ph']
        batch['f0_gt'] = denorm_f0(batch['f0'], batch['uv'], hparams)
        if use_pe:
            batch['f0_pred'] = self.pe(outputs['mel_out'])['f0_denorm_pred'].detach()
        else:
            batch['f0_pred'] = outputs.get('f0_denorm')
        return self.after_infer(batch, singer, in_path)

    @timeit
    def after_infer(self, prediction, singer, in_path):
        for k, v in prediction.items():
            if type(v) is torch.Tensor:
                prediction[k] = v.cpu().numpy()

        # remove paddings
        mel_gt = prediction["mels"]
        mel_gt_mask = np.abs(mel_gt).sum(-1) > 0

        mel_pred = prediction["outputs"]
        mel_pred_mask = np.abs(mel_pred).sum(-1) > 0
        mel_pred = mel_pred[mel_pred_mask]
        mel_pred = np.clip(mel_pred, hparams['mel_vmin'], hparams['mel_vmax'])

        f0_gt = prediction.get("f0_gt")
        f0_pred = prediction.get("f0_pred")
        if f0_pred is not None:
            f0_gt = f0_gt[mel_gt_mask]
        if len(f0_pred) > len(mel_pred_mask):
            f0_pred = f0_pred[:len(mel_pred_mask)]
        f0_pred = f0_pred[mel_pred_mask]
        torch.cuda.is_available() and torch.cuda.empty_cache()

        if singer:
            data_path = in_path.replace("batch", "singer_data")
            mel_path = data_path[:-4] + "_mel.npy"
            f0_path = data_path[:-4] + "_f0.npy"
            np.save(mel_path, mel_pred)
            np.save(f0_path, f0_pred)
        wav_pred = self.vocoder.spec2wav(mel_pred, f0=f0_pred)
        return f0_gt, f0_pred, wav_pred

    def temporary_dict2processed_input(self, item_name, temp_dict, use_crepe=True, thre=0.05):
        '''
            process data in temporary_dicts
        '''

        binarization_args = hparams['binarization_args']

        @timeit
        def get_pitch(wav, mel):
            # get ground truth f0 by self.get_pitch_algorithm
            global f0_dict
            if use_crepe:
                md5 = get_md5(wav)
                if f"{md5}_gt" in f0_dict.keys():
                    print("load temp crepe f0")
                    gt_f0 = np.array(f0_dict[f"{md5}_gt"]["f0"])
                    coarse_f0 = np.array(f0_dict[f"{md5}_coarse"]["f0"])
                else:
                    torch.cuda.is_available() and torch.cuda.empty_cache()
                    gt_f0, coarse_f0 = get_pitch_crepe(wav, mel, hparams, thre)
                f0_dict[f"{md5}_gt"] = {"f0": gt_f0.tolist(), "time": int(time.time())}
                f0_dict[f"{md5}_coarse"] = {"f0": coarse_f0.tolist(), "time": int(time.time())}
                write_temp("./infer_tools/f0_temp.json", f0_dict)
            else:
                gt_f0, coarse_f0 = get_pitch_parselmouth(wav, mel, hparams)
            processed_input['f0'] = gt_f0
            processed_input['pitch'] = coarse_f0

        def get_align(mel, phone_encoded):
            mel2ph = np.zeros([mel.shape[0]], int)
            start_frame = 0
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

    def pre(self, wav_fn, accelerate, use_crepe=True, thre=0.05):
        if isinstance(wav_fn, BytesIO):
            item_name = self.project_name
        else:
            song_info = wav_fn.split('/')
            item_name = song_info[-1].split('.')[-2]
        temp_dict = {'wav_fn': wav_fn, 'spk_id': self.project_name}

        temp_dict = self.temporary_dict2processed_input(item_name, temp_dict, use_crepe, thre)
        hparams['pndm_speedup'] = accelerate
        batch = processed_input2batch([getitem(temp_dict)])
        return batch


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
