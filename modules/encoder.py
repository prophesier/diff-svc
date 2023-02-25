import torch
from modules.commons.common_layers import *
from modules.commons.common_layers import Embedding
from modules.commons.common_layers import SinusoidalPositionalEmbedding
from utils.hparams import hparams
import numpy as np
import math


class LayerNorm(torch.nn.LayerNorm):
    """Layer normalization module.
    :param int nout: output dim size
    :param int dim: dimension to be normalized
    """

    def __init__(self, nout, dim=-1):
        """Construct an LayerNorm object."""
        super(LayerNorm, self).__init__(nout, eps=1e-12)
        self.dim = dim

    def forward(self, x):
        """Apply layer normalization.
        :param torch.Tensor x: input tensor
        :return: layer normalized tensor
        :rtype torch.Tensor
        """
        if self.dim == -1:
            return super(LayerNorm, self).forward(x)
        return super(LayerNorm, self).forward(x.transpose(1, -1)).transpose(1, -1)


class PitchPredictor(torch.nn.Module):
    def __init__(self, idim, n_layers=5, n_chans=384, odim=2, kernel_size=5,
                 dropout_rate=0.1, padding='SAME'):
        super(PitchPredictor, self).__init__()
        self.conv = torch.nn.ModuleList()
        self.kernel_size = kernel_size
        self.padding = padding
        for idx in range(n_layers):
            in_chans = idim if idx == 0 else n_chans
            self.conv += [torch.nn.Sequential(
                torch.nn.ConstantPad1d(((kernel_size - 1) // 2, (kernel_size - 1) // 2)
                                       if padding == 'SAME'
                                       else (kernel_size - 1, 0), 0),
                torch.nn.Conv1d(in_chans, n_chans, kernel_size, stride=1, padding=0),
                torch.nn.ReLU(),
                LayerNorm(n_chans, dim=1),
                torch.nn.Dropout(dropout_rate)
            )]
        self.linear = torch.nn.Linear(n_chans, odim)
        self.embed_positions = SinusoidalPositionalEmbedding(idim, 0, init_size=4096)
        self.pos_embed_alpha = nn.Parameter(torch.Tensor([1]))

    def forward(self, xs):
        positions = self.pos_embed_alpha * self.embed_positions(xs[..., 0])
        xs = xs + positions
        xs = xs.transpose(1, -1)  # (B, idim, Tmax)
        for f in self.conv:
            xs = f(xs)  # (B, C, Tmax)
        xs = self.linear(xs.transpose(1, -1))  # (B, Tmax, H)
        return xs


class SvcEncoder(nn.Module):
    def __init__(self, dictionary, out_dims=None):
        super().__init__()
        # self.dictionary = dictionary
        self.padding_idx = 0
        self.hidden_size = hparams['hidden_size']
        self.out_dims = out_dims
        if out_dims is None:
            self.out_dims = hparams['audio_num_mel_bins']
        self.mel_out = Linear(self.hidden_size, self.out_dims, bias=True)
        predictor_hidden = hparams['predictor_hidden'] if hparams['predictor_hidden'] > 0 else self.hidden_size
        if hparams['use_pitch_embed']:
            self.pitch_embed = Embedding(300, self.hidden_size, self.padding_idx)
            self.pitch_predictor = PitchPredictor(
                self.hidden_size,
                n_chans=predictor_hidden,
                n_layers=hparams['predictor_layers'],
                dropout_rate=hparams['predictor_dropout'],
                odim=2 if hparams['pitch_type'] == 'frame' else 1,
                padding=hparams['ffn_padding'], kernel_size=hparams['predictor_kernel'])
        if hparams['use_energy_embed']:
            self.energy_embed = Embedding(256, self.hidden_size, self.padding_idx)
        if hparams['use_spk_id']:
            self.spk_embed_proj = Embedding(hparams['num_spk'], self.hidden_size)
            if hparams['use_split_spk_id']:
                self.spk_embed_f0 = Embedding(hparams['num_spk'], self.hidden_size)
                self.spk_embed_dur = Embedding(hparams['num_spk'], self.hidden_size)
        elif hparams['use_spk_embed']:
            self.spk_embed_proj = Linear(256, self.hidden_size, bias=True)
        if hparams['pitch_norm'] == 'standard':
            self.pitch_norm = True
        else:
            self.pitch_norm = False
        self.f0_bin = hparams['f0_bin']
        self.f0_max = hparams['f0_max']
        self.f0_min = hparams['f0_min']

    def forward(self, hubert, mel2ph=None, spk_embed=None, f0=None):
        encoder_out = hubert
        decoder_inp = F.pad(encoder_out, [0, 0, 1, 0])
        mel2ph_ = mel2ph.unsqueeze(2).repeat([1, 1, encoder_out.shape[-1]])
        decoder_inp = torch.gather(decoder_inp, 1, mel2ph_)  # [B, T, H]
        tgt_nonpadding = (mel2ph > 0).float()[:, :, None]
        rdecoder_inp, f0_denorm, pitch_pred = self.add_pitch(f0, mel2ph)
        decoder_inp = decoder_inp + rdecoder_inp.cpu()
        decoder_inp = (decoder_inp + spk_embed) * tgt_nonpadding
        return decoder_inp.transpose(1, 2), f0_denorm

    def add_pitch(self, f0, mel2ph):
        pitch_padding = (mel2ph == 0)
        f0_denorm = self.denorm_f0(f0, pitch_padding=pitch_padding)
        f0[pitch_padding] = 0
        pitch = self.f0_to_coarse(f0_denorm)
        pitch_pred = pitch.unsqueeze(-1)
        pitch_embedding = self.pitch_embed(pitch).cuda()
        return pitch_embedding, f0_denorm, pitch_pred

    def denorm_f0(self, f0, pitch_padding=None):
        f0 = 2 ** f0
        f0[pitch_padding] = 0
        return f0

    def f0_to_coarse(self, f0):
        f0_mel_min = 1127 * math.log(1 + self.f0_min / 700)
        f0_mel_max = 1127 * math.log(1 + self.f0_max / 700)
        f0_mel = 1127 * (1 + f0 / 700).log()
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * (self.f0_bin - 2) / (f0_mel_max - f0_mel_min) + 1

        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > self.f0_bin - 1] = self.f0_bin - 1
        f0_coarse = (f0_mel + 0.5).long()
        return f0_coarse
