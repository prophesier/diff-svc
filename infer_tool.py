import numpy as np
import torch
import time
import utils
from network.vocoders.base_vocoder import VOCODERS
from preprocessing.data_gen_utils import get_pitch_parselmouth, get_pitch_crepe
from utils.hparams import hparams
from utils.pitch_utils import norm_interp_f0
from utils.pitch_utils import denorm_f0

class infer_tool:
    def __init__(self,encoder,model,pe,vocoder):
        self.encoder=encoder
        self.model=model
        self.pe=pe
        self.vocoder=vocoder
    def file2temporary_dict(self,wav_fn):
        '''
            read from file, store data in temporary dicts
        '''
        song_info = wav_fn.split('/')
        item_name = raw_item_name = song_info[-1].split('.')[-2]
        temp_dict = {}

        temp_dict['wav_fn'] = wav_fn
        temp_dict['spk_id'] = 'opencpop'

        return item_name,temp_dict
        
    def temporary_dict2processed_input(self,item_name, temp_dict,use_crepe=True,thre=0.05):
        '''
            process data in temporary_dicts
        '''
        
        binarization_args=hparams['binarization_args']
        def get_pitch(wav, mel):
            # get ground truth f0 by self.get_pitch_algorithm
            st=time.time()
            if use_crepe:
                torch.cuda.is_available() and torch.cuda.empty_cache()
                gt_f0,coarse_f0 = get_pitch_crepe(wav, mel, hparams,thre)#
            else:
                gt_f0,coarse_f0 =get_pitch_parselmouth(wav, mel, hparams)
            et=time.time()
            method='crepe' if use_crepe else 'parselmouth'
            print(f'f0 (by {method}) time used {et-st}')
            processed_input['f0'] = gt_f0
            processed_input['pitch'] = coarse_f0

        def get_align(meta_data, mel, phone_encoded, hop_size=hparams['hop_size'], audio_sample_rate=hparams['audio_sample_rate']):
            mel2ph = np.zeros([mel.shape[0]], int)
            start_frame=1
            ph_durs = mel.shape[0]/phone_encoded.shape[0]
            if hparams['debug']:
                print(mel.shape,phone_encoded.shape,mel.shape[0]/phone_encoded.shape[0])
            for i_ph in range(phone_encoded.shape[0]):
                
                end_frame = int(i_ph*ph_durs +ph_durs+ 0.5)
                mel2ph[start_frame:end_frame+1] = i_ph + 1
                start_frame = end_frame+1

            processed_input['mel2ph'] = mel2ph
        st=time.time()
        if hparams['vocoder'] in VOCODERS:
            wav, mel = VOCODERS[hparams['vocoder']].wav2spec(temp_dict['wav_fn'])
        else:
            wav, mel = VOCODERS[hparams['vocoder'].split('.')[-1]].wav2spec(temp_dict['wav_fn'])
        et=time.time()
        print(f'mel time used {et-st}')
        processed_input = {
            'item_name': item_name, 'mel': mel,
            'sec': len(wav) / hparams['audio_sample_rate'], 'len': mel.shape[0]
        }
        processed_input = {**temp_dict, **processed_input} # merge two dicts

        if binarization_args['with_f0']:
            get_pitch(wav, mel)
            
        if binarization_args['with_hubert']:
            st=time.time()
            hubert_encoded = processed_input['hubert'] = self.encoder.encode(temp_dict['wav_fn'])
            et=time.time()
            dev='cuda' if hparams['hubert_gpu'] and torch.cuda.is_available() else 'cpu'
            print(f'hubert (on {dev}) time used {et-st}')
            
            if binarization_args['with_align']:
                get_align(temp_dict, mel, hubert_encoded)

        return processed_input
    def getitem(self,item):
        max_frames = hparams['max_frames']
        spec = torch.Tensor(item['mel'])[:max_frames]
        energy = (spec.exp() ** 2).sum(-1).sqrt()
        mel2ph = torch.LongTensor(item['mel2ph'])[:max_frames] if 'mel2ph' in item else None
        f0, uv = norm_interp_f0(item["f0"][:max_frames], hparams)
        hubert=torch.Tensor(item['hubert'][:hparams['max_input_tokens']])
        pitch = torch.LongTensor(item.get("pitch"))[:max_frames]
        sample = {
            "item_name": item['item_name'],
            "hubert":hubert,
            "mel": spec,
            "pitch": pitch,
            "energy": energy,
            "f0": f0,
            "uv": uv,
            "mel2ph": mel2ph,
            "mel_nonpadding": spec.abs().sum(-1) > 0,
        }
        return sample
    def processed_input2batch(self,samples):
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
            'hubert':hubert,
            'mels': mels,
            'mel_lengths': mel_lengths,
            'mel2ph': mel2ph,
            'energy': energy,
            'pitch': pitch,
            'f0': f0,
            'uv': uv,
        }
        return batch

    def test_step(self,sample,key,use_pe=True,**kwargs):
        spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_ids')
        hubert = sample['hubert']
        mel2ph, uv, f0 = None, None, None
        ref_mels = sample["mels"]
        mel2ph = sample['mel2ph']
        sample['f0'] = sample['f0']+(key/12)
        f0 = sample['f0']
        uv = sample['uv']
        outputs = self.model(
                hubert.cuda(), spk_embed=spk_embed, mel2ph=mel2ph.cuda(), f0=f0.cuda(), uv=uv.cuda(), ref_mels=ref_mels.cuda(), infer=True,**kwargs)
        sample['outputs'] = self.model.out2mel(outputs['mel_out'])
        sample['mel2ph_pred'] = outputs['mel2ph']
        sample['f0_gt'] = denorm_f0(sample['f0'], sample['uv'], hparams)
        if use_pe:
            sample['f0_pred'] = self.pe(outputs['mel_out'])['f0_denorm_pred'].detach()#pe(ref_mels.cuda())['f0_denorm_pred'].detach()#
        else:
            sample['f0_pred'] = outputs.get('f0_denorm')
        return self.after_infer(sample)

    def after_infer(self,prediction):
        for k, v in prediction.items():
            if type(v) is torch.Tensor:
                prediction[k] = v.cpu().numpy()

        # remove paddings
        mel_gt = prediction["mels"]
        mel_gt_mask = np.abs(mel_gt).sum(-1) > 0
        mel_gt = mel_gt[mel_gt_mask]
        mel2ph_gt = prediction.get("mel2ph")
        mel2ph_gt = mel2ph_gt[mel_gt_mask] if mel2ph_gt is not None else None
        mel_pred = prediction["outputs"]
        mel_pred_mask = np.abs(mel_pred).sum(-1) > 0
        mel_pred = mel_pred[mel_pred_mask]
        mel_gt = np.clip(mel_gt, hparams['mel_vmin'], hparams['mel_vmax'])
        mel_pred = np.clip(mel_pred, hparams['mel_vmin'], hparams['mel_vmax'])

        mel2ph_pred = prediction.get("mel2ph_pred")
        if mel2ph_pred is not None:
            if len(mel2ph_pred) > len(mel_pred_mask):
                mel2ph_pred = mel2ph_pred[:len(mel_pred_mask)]
            mel2ph_pred = mel2ph_pred[mel_pred_mask]

        f0_gt = prediction.get("f0_gt")
        f0_pred = prediction.get("f0_pred")
        if f0_pred is not None:
            f0_gt = f0_gt[mel_gt_mask]
        if len(f0_pred) > len(mel_pred_mask):
            f0_pred = f0_pred[:len(mel_pred_mask)]
        f0_pred = f0_pred[mel_pred_mask]
        torch.cuda.is_available() and torch.cuda.empty_cache()
        wav_pred = self.vocoder.spec2wav(mel_pred, f0=f0_pred)
        return f0_gt,f0_pred,wav_pred
    
    def make_dict(self,wav_fn,use_crepe=True,thre=0.05):
        return self.temporary_dict2processed_input(*self.file2temporary_dict(wav_fn),use_crepe=use_crepe,thre=thre)
    
    def infer(self,tempdict,key=0,use_pe=True,use_gt_mel=False,add_noise_step=1000):
        batch=self.processed_input2batch([self.getitem(tempdict)])
        return self.test_step(batch,key=key,use_pe=use_pe,use_gt_mel=use_gt_mel,add_noise_step=add_noise_step)