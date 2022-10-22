from modules.commons.common_layers import *
from modules.commons.common_layers import Embedding
from modules.fastspeech.tts_modules import FastspeechDecoder, DurationPredictor, LengthRegulator, PitchPredictor, \
    EnergyPredictor, FastspeechEncoder
from utils.cwt import cwt2f0
from utils.hparams import hparams
from utils.pitch_utils import f0_to_coarse, denorm_f0, norm_f0
from modules.fastspeech.fs2 import FastSpeech2
from utils.text_encoder import TokenTextEncoder
from tts.data_gen.txt_processors.zh_g2pM import ALL_YUNMU
from torch.nn import functional as F
import torch
from training.diffsinger import Batch2Loss


class FastspeechMIDIEncoder(FastspeechEncoder):
    '''
        compared to FastspeechEncoder:
        - adds new input items (midi, midi_dur, slur)

        but these are same:
        - positional encoding
    '''
    def forward_embedding(self, txt_tokens, midi_embedding, midi_dur_embedding, slur_embedding):
        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(txt_tokens)
        x = x + midi_embedding + midi_dur_embedding + slur_embedding
        if hparams['use_pos_embed']:
            if hparams.get('rel_pos') is not None and hparams['rel_pos']:
                x = self.embed_positions(x)
            else:
                positions = self.embed_positions(txt_tokens)
                x = x + positions
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self, txt_tokens, midi_embedding, midi_dur_embedding, slur_embedding):
        """
        :param txt_tokens: [B, T]
        :return: {
            'encoder_out': [T x B x C]
        }
        """
        encoder_padding_mask = txt_tokens.eq(self.padding_idx).detach()
        x = self.forward_embedding(txt_tokens, midi_embedding, midi_dur_embedding, slur_embedding)  # [B, T, H]
        x = super(FastspeechEncoder, self).forward(x, encoder_padding_mask)
        return x


FS_ENCODERS = {
    'fft': lambda hp, embed_tokens, d: FastspeechMIDIEncoder(
        embed_tokens, hp['hidden_size'], hp['enc_layers'], hp['enc_ffn_kernel_size'],
        num_heads=hp['num_heads']),
}


class FastSpeech2MIDI(FastSpeech2):
    def __init__(self, dictionary, out_dims=None):
        super().__init__(dictionary, out_dims)
        del self.encoder
        
        self.encoder = FS_ENCODERS[hparams['encoder_type']](hparams, self.encoder_embed_tokens, self.dictionary)
        self.midi_embed = Embedding(300, self.hidden_size, self.padding_idx)
        self.midi_dur_layer = Linear(1, self.hidden_size)
        self.is_slur_embed = Embedding(2, self.hidden_size)
        yunmu = ['AP', 'SP'] + ALL_YUNMU 
        yunmu.remove('ng')
        self.vowel_tokens = [dictionary.encode(ph)[0] for ph in yunmu]

    def forward(self, txt_tokens, mel2ph=None, spk_embed_id=None,
                ref_mels=None, f0=None, uv=None, energy=None, skip_decoder=False,
                spk_embed_dur_id=None, spk_embed_f0_id=None, infer=False, **kwargs):
        '''
            steps:
            1. embedding: midi_embedding, midi_dur_embedding, slur_embedding
            2. run self.encoder (a Transformer) using txt_tokens and embeddings
            3. run *dur_predictor* in *add_dur* using *encoder_out*, get *ret['dur']* and *ret['mel2ph']*
            4. the same for *pitch_predictor* and *energy_predictor*
            5. run decoder (skipped for diffusion)
        '''
        midi_embedding, midi_dur_embedding, slur_embedding = Batch2Loss.insert1(
            kwargs['pitch_midi'], kwargs.get('midi_dur', None), kwargs.get('is_slur', None),
            self.midi_embed, self.midi_dur_layer, self.is_slur_embed
        )

        encoder_out = Batch2Loss.module1(self.encoder, txt_tokens, midi_embedding, midi_dur_embedding, slur_embedding)  # [B, T, C]
        
        src_nonpadding = (txt_tokens > 0).float()[:, :, None]
        var_embed, spk_embed, spk_embed_dur, spk_embed_f0, dur_inp = Batch2Loss.insert2(
            encoder_out, spk_embed_id, spk_embed_dur_id, spk_embed_f0_id, src_nonpadding,
            self.spk_embed_proj if hasattr(self, 'spk_embed_proj') else None
        )

        ret = {}
        mel2ph = Batch2Loss.module2(
            self.dur_predictor, self.length_regulator,
            dur_inp, mel2ph, txt_tokens, self.vowel_tokens, ret, midi_dur=kwargs['midi_dur']*hparams['audio_sample_rate']/hparams['hop_size']
        )

        tgt_nonpadding = (mel2ph > 0).float()[:, :, None]
        decoder_inp, pitch_inp, pitch_inp_ph = Batch2Loss.insert3(
            encoder_out, mel2ph, var_embed, spk_embed_f0, src_nonpadding, tgt_nonpadding
        )

        pitch_embedding, energy_embedding = Batch2Loss.module3(
            getattr(self, 'pitch_predictor', None), getattr(self, 'pitch_embed', None),
            getattr(self, 'energy_predictor', None), getattr(self, 'energy_embed', None),
            pitch_inp, pitch_inp_ph, f0, uv, energy, mel2ph, (not infer), ret
        )

        decoder_inp = Batch2Loss.insert4(
            decoder_inp, pitch_embedding, energy_embedding, spk_embed, ret, tgt_nonpadding
        )

        if skip_decoder:
            return ret
        ret['mel_out'] = self.run_decoder(decoder_inp, tgt_nonpadding, ret, infer=infer, **kwargs)

        return ret