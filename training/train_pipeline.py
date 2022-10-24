from utils.hparams import hparams
import torch
from torch.nn import functional as F
from utils.pitch_utils import f0_to_coarse, denorm_f0, norm_f0

class Batch2Loss:
    '''
        pipeline: batch -> insert1 -> module1 -> insert2 -> module2 -> insert3 -> module3 -> insert4 -> module4 -> loss
    '''

    @staticmethod
    def insert1(pitch_midi, midi_dur, is_slur, # variables
                midi_embed, midi_dur_layer, is_slur_embed): # modules
        '''
            add embeddings for midi, midi_dur, slur
        '''
        midi_embedding = midi_embed(pitch_midi)
        midi_dur_embedding, slur_embedding = 0, 0
        if midi_dur is not None:
            midi_dur_embedding = midi_dur_layer(midi_dur[:, :, None])  # [B, T, 1] -> [B, T, H]
        if is_slur is not None:
            slur_embedding = is_slur_embed(is_slur)
        return midi_embedding, midi_dur_embedding, slur_embedding

    @staticmethod
    def module1(fs2_encoder, # modules
                txt_tokens, midi_embedding, midi_dur_embedding, slur_embedding): # variables
        '''
            get *encoder_out* == fs2_encoder(*txt_tokens*, some embeddings)
        '''
        encoder_out = fs2_encoder(txt_tokens, midi_embedding, midi_dur_embedding, slur_embedding)
        return encoder_out

    @staticmethod
    def insert2(encoder_out, spk_embed_id, spk_embed_dur_id, spk_embed_f0_id, src_nonpadding, # variables
                spk_embed_proj): # modules
        '''
            1. add embeddings for pspk, spk_dur, sk_f0
            2. get *dur_inp* ~= *encoder_out* + *spk_embed_dur*
        '''
        # add ref style embed
        # Not implemented
        # variance encoder
        var_embed = 0

        # encoder_out_dur denotes encoder outputs for duration predictor
        # in speech adaptation, duration predictor use old speaker embedding
        if hparams['use_spk_embed']:
            spk_embed_dur = spk_embed_f0 = spk_embed = spk_embed_proj(spk_embed_id)[:, None, :]
        elif hparams['use_spk_id']:
            if spk_embed_dur_id is None:
                spk_embed_dur_id = spk_embed_id
            if spk_embed_f0_id is None:
                spk_embed_f0_id = spk_embed_id
            spk_embed = spk_embed_proj(spk_embed_id)[:, None, :]
            spk_embed_dur = spk_embed_f0 = spk_embed
            if hparams['use_split_spk_id']:
                spk_embed_dur = spk_embed_dur(spk_embed_dur_id)[:, None, :]
                spk_embed_f0 = spk_embed_f0(spk_embed_f0_id)[:, None, :]
        else:
            spk_embed_dur = spk_embed_f0 = spk_embed = 0

        # add dur
        dur_inp = (encoder_out + var_embed + spk_embed_dur) * src_nonpadding
        return var_embed, spk_embed, spk_embed_dur, spk_embed_f0, dur_inp

    @staticmethod
    def module2(dur_predictor, length_regulator, # modules
                dur_input, mel2ph, txt_tokens, all_vowel_tokens, ret, midi_dur=None): # variables
        '''
            1. get *dur* ~= dur_predictor(*dur_inp*)
            2. (mel2ph is None): get *mel2ph* ~= length_regulater(*dur*)
        ''' 
        src_padding = (txt_tokens == 0)
        dur_input = dur_input.detach() + hparams['predictor_grad'] * (dur_input - dur_input.detach())
        
        if mel2ph is None:
            dur, xs = dur_predictor.inference(dur_input, src_padding)
            ret['dur'] = xs
            dur = xs.squeeze(-1).exp() - 1.0
            for i in range(len(dur)):
                for j in range(len(dur[i])):
                    if txt_tokens[i,j] in all_vowel_tokens:
                        if j < len(dur[i])-1 and txt_tokens[i,j+1] not in all_vowel_tokens:
                            dur[i,j] = midi_dur[i,j] - dur[i,j+1]
                            if dur[i,j] < 0:
                                dur[i,j] = 0
                                dur[i,j+1] = midi_dur[i,j]
                        else:
                            dur[i,j]=midi_dur[i,j]      
            dur[:,0] = dur[:,0] + 0.5
            dur_acc = F.pad(torch.round(torch.cumsum(dur, axis=1)), (1,0))
            dur = torch.clamp(dur_acc[:,1:]-dur_acc[:,:-1], min=0).long()
            ret['dur_choice'] = dur
            mel2ph = length_regulator(dur, src_padding).detach()
        else:
            ret['dur'] = dur_predictor(dur_input, src_padding)
        ret['mel2ph'] = mel2ph

        return mel2ph
    
    @staticmethod
    def insert3(encoder_out, mel2ph, var_embed, spk_embed_f0, src_nonpadding, tgt_nonpadding): # variables
        '''
            1. get *decoder_inp* ~= gather *encoder_out* according to *mel2ph*
            2. get *pitch_inp* ~= *decoder_inp* + *spk_embed_f0*
            3. get *pitch_inp_ph* ~= *encoder_out* + *spk_embed_f0*
        '''
        decoder_inp = F.pad(encoder_out, [0, 0, 1, 0])
        mel2ph_ = mel2ph[..., None].repeat([1, 1, encoder_out.shape[-1]])
        decoder_inp = decoder_inp_origin = torch.gather(decoder_inp, 1, mel2ph_)  # [B, T, H]

        pitch_inp = (decoder_inp_origin + var_embed + spk_embed_f0) * tgt_nonpadding
        pitch_inp_ph = (encoder_out + var_embed + spk_embed_f0) * src_nonpadding
        return decoder_inp, pitch_inp, pitch_inp_ph

    @staticmethod
    def module3(pitch_predictor, pitch_embed, energy_predictor, energy_embed, # modules
                pitch_inp, pitch_inp_ph, f0, uv, energy, mel2ph, is_training, ret): # variables
        '''
            1. get *ret['pitch_pred']*, *ret['energy_pred']* ~= pitch_predictor(*pitch_inp*), energy_predictor(*pitch_inp*)
            2. get *pitch_embedding* ~= pitch_embed(f0_to_coarse(denorm_f0(*f0* or *pitch_pred*))
            3. get *energy_embedding* ~= energy_embed(energy_to_coarse(*energy* or *energy_pred*))
        '''
        def add_pitch(decoder_inp, f0, uv, mel2ph, ret, encoder_out=None):
            if hparams['pitch_type'] == 'ph':
                pitch_pred_inp = encoder_out.detach() + hparams['predictor_grad'] * (encoder_out - encoder_out.detach())
                pitch_padding = (encoder_out.sum().abs() == 0)
                ret['pitch_pred'] = pitch_pred = pitch_predictor(pitch_pred_inp)
                if f0 is None:
                    f0 = pitch_pred[:, :, 0]
                ret['f0_denorm'] = f0_denorm = denorm_f0(f0, None, hparams, pitch_padding=pitch_padding)
                pitch = f0_to_coarse(f0_denorm)  # start from 0 [B, T_txt]
                pitch = F.pad(pitch, [1, 0])
                pitch = torch.gather(pitch, 1, mel2ph)  # [B, T_mel]
                pitch_embedding = pitch_embed(pitch)
                return pitch_embedding
            
            decoder_inp = decoder_inp.detach() + hparams['predictor_grad'] * (decoder_inp - decoder_inp.detach())

            pitch_padding = (mel2ph == 0)

            if hparams['pitch_type'] == 'cwt':
                # NOTE: this part of script is *isolated* from other scripts, which means
                #       it may not be compatible with the current version.    
                pass
                # pitch_padding = None
                # ret['cwt'] = cwt_out = self.cwt_predictor(decoder_inp)
                # stats_out = self.cwt_stats_layers(encoder_out[:, 0, :])  # [B, 2]
                # mean = ret['f0_mean'] = stats_out[:, 0]
                # std = ret['f0_std'] = stats_out[:, 1]
                # cwt_spec = cwt_out[:, :, :10]
                # if f0 is None:
                #     std = std * hparams['cwt_std_scale']
                #     f0 = self.cwt2f0_norm(cwt_spec, mean, std, mel2ph)
                #     if hparams['use_uv']:
                #         assert cwt_out.shape[-1] == 11
                #         uv = cwt_out[:, :, -1] > 0
            elif hparams['pitch_ar']:
                ret['pitch_pred'] = pitch_pred = pitch_predictor(decoder_inp, f0 if is_training else None)
                if f0 is None:
                    f0 = pitch_pred[:, :, 0]
            else:
                ret['pitch_pred'] = pitch_pred = pitch_predictor(decoder_inp)
                if f0 is None:
                    f0 = pitch_pred[:, :, 0]
                if hparams['use_uv'] and uv is None:
                    uv = pitch_pred[:, :, 1] > 0
            ret['f0_denorm'] = f0_denorm = denorm_f0(f0, uv, hparams, pitch_padding=pitch_padding)
            if pitch_padding is not None:
                f0[pitch_padding] = 0

            pitch = f0_to_coarse(f0_denorm)  # start from 0
            pitch_embedding = pitch_embed(pitch)
            return pitch_embedding

        def add_energy(decoder_inp, energy, ret):
            decoder_inp = decoder_inp.detach() + hparams['predictor_grad'] * (decoder_inp - decoder_inp.detach())
            ret['energy_pred'] = energy_pred = energy_predictor(decoder_inp)[:, :, 0]
            if energy is None:
                energy = energy_pred
            energy = torch.clamp(energy * 256 // 4, max=255).long() # energy_to_coarse
            energy_embedding = energy_embed(energy)
            return energy_embedding

        # add pitch and energy embed
        nframes = mel2ph.size(1)

        pitch_embedding = 0
        if hparams['use_pitch_embed']:
            if f0 is not None:
                delta_l = nframes - f0.size(1)
                if delta_l > 0:
                    f0 = torch.cat((f0,torch.FloatTensor([[x[-1]] * delta_l for x in f0]).to(f0.device)),1)
                f0 = f0[:,:nframes]
            if uv is not None:
                delta_l = nframes - uv.size(1)
                if delta_l > 0:
                    uv = torch.cat((uv,torch.FloatTensor([[x[-1]] * delta_l for x in uv]).to(uv.device)),1)
                uv = uv[:,:nframes]
            pitch_embedding = add_pitch(pitch_inp, f0, uv, mel2ph, ret, encoder_out=pitch_inp_ph)
           
        energy_embedding = 0
        if hparams['use_energy_embed']:
            if energy is not None:
                delta_l = nframes - energy.size(1)
                if delta_l > 0:
                    energy = torch.cat((energy,torch.FloatTensor([[x[-1]] * delta_l for x in energy]).to(energy.device)),1)
                energy = energy[:,:nframes]
            energy_embedding = add_energy(pitch_inp, energy, ret)
        
        return pitch_embedding, energy_embedding
    
    @staticmethod
    def insert4(decoder_inp, pitch_embedding, energy_embedding, spk_embed, ret, tgt_nonpadding):
        '''
            *decoder_inp* ~= *decoder_inp* + embeddings for spk, pitch, energy
        '''
        ret['decoder_inp'] = decoder_inp = (decoder_inp + pitch_embedding + energy_embedding + spk_embed) * tgt_nonpadding
        return decoder_inp

    @staticmethod
    def module4(diff_main_loss, # modules
                norm_spec, decoder_inp_t, ret, K_step, batch_size, device): # variables
        '''
            training diffusion using spec as input and decoder_inp as condition.
            
            Args:
                norm_spec: (normalized) spec
                decoder_inp_t: (transposed) decoder_inp
            Returns:
                ret['diff_loss']
        '''
        t = torch.randint(0, K_step, (batch_size,), device=device).long()
        norm_spec = norm_spec.transpose(1, 2)[:, None, :, :]  # [B, 1, M, T]
        ret['diff_loss'] = diff_main_loss(norm_spec, t, cond=decoder_inp_t)
        # nonpadding = (mel2ph != 0).float()
        # ret['diff_loss'] = self.p_losses(x, t, cond, nonpadding=nonpadding)
