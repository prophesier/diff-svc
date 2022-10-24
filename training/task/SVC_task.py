import torch

import utils
from utils.hparams import hparams
from network.diff.net import DiffNet
from network.diff.diffusion import GaussianDiffusion, OfflineGaussianDiffusion
from training.task.fs2 import FastSpeech2Task
from network.vocoders.base_vocoder import get_vocoder_cls, BaseVocoder
from modules.fastspeech.tts_modules import mel2ph_to_dur

from network.diff.candidate_decoder import FFT
from utils.pitch_utils import denorm_f0
from training.dataset.fs2_utils import FastSpeechDataset

import numpy as np
import os
import torch.nn.functional as F

DIFF_DECODERS = {
    'wavenet': lambda hp: DiffNet(hp['audio_num_mel_bins']),
    'fft': lambda hp: FFT(
        hp['hidden_size'], hp['dec_layers'], hp['dec_ffn_kernel_size'], hp['num_heads']),
}


class SVCDataset(FastSpeechDataset):
    def collater(self, samples):
        from preprocessing.process_pipeline import File2Batch
        return File2Batch.processed_input2batch(samples)


class SVCTask(FastSpeech2Task):
    def __init__(self):
        super(SVCTask, self).__init__()
        self.dataset_cls = SVCDataset
        self.vocoder: BaseVocoder = get_vocoder_cls(hparams)()
    
    def build_tts_model(self):
        # import torch
        # from tqdm import tqdm
        # v_min = torch.ones([80]) * 100
        # v_max = torch.ones([80]) * -100
        # for i, ds in enumerate(tqdm(self.dataset_cls('train'))):
        #     v_max = torch.max(torch.max(ds['mel'].reshape(-1, 80), 0)[0], v_max)
        #     v_min = torch.min(torch.min(ds['mel'].reshape(-1, 80), 0)[0], v_min)
        #     if i % 100 == 0:
        #         print(i, v_min, v_max)
        # print('final', v_min, v_max)
        mel_bins = hparams['audio_num_mel_bins']
        self.model = GaussianDiffusion(
            phone_encoder=self.phone_encoder,
            out_dims=mel_bins, denoise_fn=DIFF_DECODERS[hparams['diff_decoder_type']](hparams),
            timesteps=hparams['timesteps'],
            K_step=hparams['K_step'],
            loss_type=hparams['diff_loss_type'],
            spec_min=hparams['spec_min'], spec_max=hparams['spec_max'],
        )

    
    def build_optimizer(self, model):
        self.optimizer = optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=hparams['lr'],
            betas=(hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']),
            weight_decay=hparams['weight_decay'])
        return optimizer

    def run_model(self, model, sample, return_output=False, infer=False):
        '''
            steps:
            1. run the full model, calc the main loss
            2. calculate loss for dur_predictor, pitch_predictor, energy_predictor
        '''
        hubert = sample['hubert']  # [B, T_t,H]
        target = sample['mels']  # [B, T_s, 80]
        mel2ph = sample['mel2ph'] # [B, T_s]
        f0 = sample['f0']
        uv = sample['uv']
        energy = sample['energy']

        spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_ids')
        if hparams['pitch_type'] == 'cwt':
            # NOTE: this part of script is *isolated* from other scripts, which means
            #       it may not be compatible with the current version.    
            pass
            # cwt_spec = sample[f'cwt_spec']
            # f0_mean = sample['f0_mean']
            # f0_std = sample['f0_std']
            # sample['f0_cwt'] = f0 = model.cwt2f0_norm(cwt_spec, f0_mean, f0_std, mel2ph)

        # output == ret
        # model == src.diff.diffusion.GaussianDiffusion
        output = model(hubert, mel2ph=mel2ph, spk_embed=spk_embed,
                       ref_mels=target, f0=f0, uv=uv, energy=energy, infer=infer)

        losses = {}
        if 'diff_loss' in output:
            losses['mel'] = output['diff_loss']
        #self.add_dur_loss(output['dur'], mel2ph, txt_tokens, sample['word_boundary'], losses=losses)
        # if hparams['use_pitch_embed']:
        #     self.add_pitch_loss(output, sample, losses)
        # if hparams['use_energy_embed']:
        #     self.add_energy_loss(output['energy_pred'], energy, losses)
        if not return_output:
            return losses
        else:
            return losses, output
    
    def _training_step(self, sample, batch_idx, _):
        log_outputs = self.run_model(self.model, sample)
        total_loss = sum([v for v in log_outputs.values() if isinstance(v, torch.Tensor) and v.requires_grad])
        log_outputs['batch_size'] = sample['hubert'].size()[0]
        log_outputs['lr'] = self.scheduler.get_lr()[0]
        return total_loss, log_outputs

    def build_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.StepLR(optimizer, hparams['decay_steps'], gamma=0.5)

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx):
        if optimizer is None:
            return
        optimizer.step()
        optimizer.zero_grad()
        if self.scheduler is not None:
            self.scheduler.step(self.global_step // hparams['accumulate_grad_batches'])

    def validation_step(self, sample, batch_idx):
        outputs = {}
        hubert = sample['hubert']  # [B, T_t]

        target = sample['mels']  # [B, T_s, 80]
        energy = sample['energy']
        # fs2_mel = sample['fs2_mels']
        spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_ids')
        mel2ph = sample['mel2ph']

        outputs['losses'] = {}

        outputs['losses'], model_out = self.run_model(self.model, sample, return_output=True, infer=False)

        outputs['total_loss'] = sum(outputs['losses'].values())
        outputs['nsamples'] = sample['nsamples']
        outputs = utils.tensors_to_scalars(outputs)
        if batch_idx < hparams['num_valid_plots']:
            model_out = self.model(
                hubert, spk_embed=spk_embed, mel2ph=mel2ph, f0=sample['f0'], uv=sample['uv'], energy=energy, ref_mels=None, infer=True
                )

            if hparams.get('pe_enable') is not None and hparams['pe_enable']:
                gt_f0 = self.pe(sample['mels'])['f0_denorm_pred']  # pe predict from GT mel
                pred_f0 = self.pe(model_out['mel_out'])['f0_denorm_pred']  # pe predict from Pred mel
            else:
                gt_f0 = denorm_f0(sample['f0'], sample['uv'], hparams)
                pred_f0 = model_out.get('f0_denorm')
            self.plot_wav(batch_idx, sample['mels'], model_out['mel_out'], is_mel=True, gt_f0=gt_f0, f0=pred_f0)
            self.plot_mel(batch_idx, sample['mels'], model_out['mel_out'], name=f'diffmel_{batch_idx}')
            #self.plot_mel(batch_idx, sample['mels'], model_out['fs2_mel'], name=f'fs2mel_{batch_idx}')
            if hparams['use_pitch_embed']:
                self.plot_pitch(batch_idx, sample, model_out)
        return outputs

    def add_dur_loss(self, dur_pred, mel2ph, txt_tokens, wdb, losses=None):
        """
            the effect of each loss component:
                hparams['dur_loss'] : align each phoneme
                hparams['lambda_word_dur']: align each word
                hparams['lambda_sent_dur']: align each sentence
            
            :param dur_pred: [B, T], float, log scale
            :param mel2ph: [B, T]
            :param txt_tokens: [B, T]
            :param losses:
            :return:
        """
        B, T = txt_tokens.shape
        nonpadding = (txt_tokens != 0).float()
        dur_gt = mel2ph_to_dur(mel2ph, T).float() * nonpadding
        is_sil = torch.zeros_like(txt_tokens).bool()
        for p in self.sil_ph:
            is_sil = is_sil | (txt_tokens == self.phone_encoder.encode(p)[0])
        is_sil = is_sil.float()  # [B, T_txt]

        # phone duration loss
        if hparams['dur_loss'] == 'mse':
            losses['pdur'] = F.mse_loss(dur_pred, (dur_gt + 1).log(), reduction='none')
            losses['pdur'] = (losses['pdur'] * nonpadding).sum() / nonpadding.sum()
            losses['pdur'] = losses['pdur'] * hparams['lambda_ph_dur']
            dur_pred = (dur_pred.exp() - 1).clamp(min=0)
        else:
            raise NotImplementedError

        # use linear scale for sent and word duration
        if hparams['lambda_word_dur'] > 0:
            #idx = F.pad(wdb.cumsum(axis=1), (1, 0))[:, :-1]
            idx = wdb.cumsum(axis=1)
            # word_dur_g = dur_gt.new_zeros([B, idx.max() + 1]).scatter_(1, idx, midi_dur)  # midi_dur can be implied by add gt-ph_dur
            word_dur_p = dur_pred.new_zeros([B, idx.max() + 1]).scatter_add(1, idx, dur_pred)
            word_dur_g = dur_gt.new_zeros([B, idx.max() + 1]).scatter_add(1, idx, dur_gt)
            wdur_loss = F.mse_loss((word_dur_p + 1).log(), (word_dur_g + 1).log(), reduction='none')
            word_nonpadding = (word_dur_g > 0).float()
            wdur_loss = (wdur_loss * word_nonpadding).sum() / word_nonpadding.sum()
            losses['wdur'] = wdur_loss * hparams['lambda_word_dur']
        if hparams['lambda_sent_dur'] > 0:
            sent_dur_p = dur_pred.sum(-1)
            sent_dur_g = dur_gt.sum(-1)
            sdur_loss = F.mse_loss((sent_dur_p + 1).log(), (sent_dur_g + 1).log(), reduction='mean')
            losses['sdur'] = sdur_loss.mean() * hparams['lambda_sent_dur']
    
    ############
    # validation plots
    ############
    def plot_wav(self, batch_idx, gt_wav, wav_out, is_mel=False, gt_f0=None, f0=None, name=None):
        gt_wav = gt_wav[0].cpu().numpy()
        wav_out = wav_out[0].cpu().numpy()
        gt_f0 = gt_f0[0].cpu().numpy()
        f0 = f0[0].cpu().numpy()
        if is_mel:
            gt_wav = self.vocoder.spec2wav(gt_wav, f0=gt_f0)
            wav_out = self.vocoder.spec2wav(wav_out, f0=f0)
        self.logger.experiment.add_audio(f'gt_{batch_idx}', gt_wav, sample_rate=hparams['audio_sample_rate'], global_step=self.global_step)
        self.logger.experiment.add_audio(f'wav_{batch_idx}', wav_out, sample_rate=hparams['audio_sample_rate'], global_step=self.global_step)


