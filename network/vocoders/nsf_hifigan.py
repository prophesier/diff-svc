import os
import torch
from modules.nsf_hifigan.models import load_model, Generator
from modules.nsf_hifigan.nvSTFT import load_wav_to_torch, STFT
from utils.hparams import hparams
from network.vocoders.base_vocoder import BaseVocoder, register_vocoder

@register_vocoder
class NsfHifiGAN(BaseVocoder):
    def __init__(self, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        model_path = hparams['vocoder_ckpt']
        if os.path.exists(model_path):
            print('| Load HifiGAN: ', model_path)
            self.model, self.h = load_model(model_path, device=self.device)
        else:
            print('Error: HifiGAN model file is not found!')

    def spec2wav_torch(self, mel, **kwargs): # mel: [B, T, bins]
        if self.h.sampling_rate != hparams['audio_sample_rate']:
            print('Mismatch parameters: hparams[\'audio_sample_rate\']=',hparams['audio_sample_rate'],'!=',self.h.sampling_rate,'(vocoder)')
        if self.h.num_mels != hparams['audio_num_mel_bins']:
            print('Mismatch parameters: hparams[\'audio_num_mel_bins\']=',hparams['audio_num_mel_bins'],'!=',self.h.num_mels,'(vocoder)')
        if self.h.n_fft != hparams['fft_size']:
            print('Mismatch parameters: hparams[\'fft_size\']=',hparams['fft_size'],'!=',self.h.n_fft,'(vocoder)')
        if self.h.win_size != hparams['win_size']:
            print('Mismatch parameters: hparams[\'win_size\']=',hparams['win_size'],'!=',self.h.win_size,'(vocoder)')
        if self.h.hop_size != hparams['hop_size']:
            print('Mismatch parameters: hparams[\'hop_size\']=',hparams['hop_size'],'!=',self.h.hop_size,'(vocoder)')
        if self.h.fmin != hparams['fmin']:
            print('Mismatch parameters: hparams[\'fmin\']=',hparams['fmin'],'!=',self.h.fmin,'(vocoder)')
        if self.h.fmax != hparams['fmax']:
            print('Mismatch parameters: hparams[\'fmax\']=',hparams['fmax'],'!=',self.h.fmax,'(vocoder)')
        with torch.no_grad():
            c = mel.transpose(2, 1) #[B, T, bins]
            #log10 to log mel
            c = 2.30259 * c
            f0 = kwargs.get('f0') #[B, T]
            if f0 is not None and hparams.get('use_nsf'):
                y = self.model(c, f0).view(-1)
            else:
                y = self.model(c).view(-1)
        return y

    def spec2wav(self, mel, **kwargs):
        if self.h.sampling_rate != hparams['audio_sample_rate']:
            print('Mismatch parameters: hparams[\'audio_sample_rate\']=',hparams['audio_sample_rate'],'!=',self.h.sampling_rate,'(vocoder)')
        if self.h.num_mels != hparams['audio_num_mel_bins']:
            print('Mismatch parameters: hparams[\'audio_num_mel_bins\']=',hparams['audio_num_mel_bins'],'!=',self.h.num_mels,'(vocoder)')
        if self.h.n_fft != hparams['fft_size']:
            print('Mismatch parameters: hparams[\'fft_size\']=',hparams['fft_size'],'!=',self.h.n_fft,'(vocoder)')
        if self.h.win_size != hparams['win_size']:
            print('Mismatch parameters: hparams[\'win_size\']=',hparams['win_size'],'!=',self.h.win_size,'(vocoder)')
        if self.h.hop_size != hparams['hop_size']:
            print('Mismatch parameters: hparams[\'hop_size\']=',hparams['hop_size'],'!=',self.h.hop_size,'(vocoder)')
        if self.h.fmin != hparams['fmin']:
            print('Mismatch parameters: hparams[\'fmin\']=',hparams['fmin'],'!=',self.h.fmin,'(vocoder)')
        if self.h.fmax != hparams['fmax']:
            print('Mismatch parameters: hparams[\'fmax\']=',hparams['fmax'],'!=',self.h.fmax,'(vocoder)')
        with torch.no_grad():
            c = torch.FloatTensor(mel).unsqueeze(0).transpose(2, 1).to(self.device)
            #log10 to log mel
            c = 2.30259 * c
            f0 = kwargs.get('f0')
            if f0 is not None and hparams.get('use_nsf'):
                f0 = torch.FloatTensor(f0[None, :]).to(self.device)
                y = self.model(c, f0).view(-1)
            else:
                y = self.model(c).view(-1)
        wav_out = y.cpu().numpy()
        return wav_out

    @staticmethod
    def wav2spec(inp_path, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        sampling_rate = hparams['audio_sample_rate']
        num_mels = hparams['audio_num_mel_bins']
        n_fft = hparams['fft_size']
        win_size =hparams['win_size']
        hop_size = hparams['hop_size']
        fmin = hparams['fmin']
        fmax = hparams['fmax']
        stft = STFT(sampling_rate, num_mels, n_fft, win_size, hop_size, fmin, fmax)
        with torch.no_grad():
            wav_torch, _ = load_wav_to_torch(inp_path, target_sr=stft.target_sr)
            mel_torch = stft.get_mel(wav_torch.unsqueeze(0).to(device)).squeeze(0).T
            #log mel to log10 mel
            mel_torch = 0.434294 * mel_torch
            return wav_torch.cpu().numpy(), mel_torch.cpu().numpy()