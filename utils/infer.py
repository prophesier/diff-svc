def get_pitch_crepe(wav_data, mel, hparams, threshold=0.3):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda")
    #crepe只支持16khz采样率，需要重采样
    wav16k = resampy.resample(wav_data, hparams['audio_sample_rate'], 16000)
    wav16k_torch = torch.FloatTensor(wav16k).unsqueeze(0).to(device)
    
    #频率范围
    f0_min = 50
    f0_max = 800
    
    #重采样后按照hopsize=80,也就是5ms一帧分析f0    
    f0, pd = torchcrepe.predict(wav16k_torch, 16000, 80, f0_min, f0_max, pad=True, model='full', batch_size=1024, device=device, return_periodicity=True)
    
    #滤波，去掉静音，设置uv阈值，参考原仓库readme
    pd = torchcrepe.filter.median(pd, 3)
    pd = torchcrepe.threshold.Silence(-60.)(pd, wav16k_torch, 16000, 80)
    f0 = torchcrepe.threshold.At(threshold)(f0, pd)
    f0 = torchcrepe.filter.mean(f0, 3)
    
    #将nan频率（uv部分）转换为0频率
    f0 = torch.where(torch.isnan(f0), torch.full_like(f0,0), f0)
    
    '''
    np.savetxt('问棋-crepe.csv',np.array([0.005*np.arange(len(f0[0])),f0[0].cpu().numpy()]).transpose(),delimiter=',')
    '''
    
    #去掉0频率，并线性插值
    nzindex = torch.nonzero(f0[0]).squeeze()
    f0 = torch.index_select(f0[0],dim=0, index=nzindex).cpu().numpy()
    time_org = 0.005*nzindex.cpu().numpy()
    time_frame = np.arange(len(mel))*hparams['hop_size']/hparams['audio_sample_rate']
    f0 = np.interp(time_frame, time_org, f0, left=f0[0], right=f0[-1])
    return f0

def file2temporary_dict(wav_fn):
    '''
        read from file, store data in temporary dicts
    '''
    song_info = wav_fn.split('/')
    item_name = raw_item_name = song_info[-1].split('.')[-2]
    temp_dict = {}

    temp_dict['wav_fn'] = wav_fn
    temp_dict['spk_id'] = 'opencpop'

    return item_name,temp_dict
    
def temporary_dict2processed_input(item_name, temp_dict,use_crepe=True,thre=0.05):
    '''
        process data in temporary_dicts
    '''
    encoder=Hubertencoder(hparams['hubert_path'])
    binarization_args=hparams['binarization_args']
    def get_pitch(wav, mel):
        # get ground truth f0 by self.get_pitch_algorithm
        if use_crepe:
            gt_f0 = get_pitch_crepe(wav, mel, hparams,thre)#
            processed_input['f0'] = gt_f0
            processed_input['pitch'] = f0_to_coarse(gt_f0,hparams)
        else:
            gt_f0,coarse_f0 =get_pitch_parselmouth(wav, mel, hparams)
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

    if hparams['vocoder'] in VOCODERS:
        wav, mel = VOCODERS[hparams['vocoder']].wav2spec(temp_dict['wav_fn'])
    else:
        wav, mel = VOCODERS[hparams['vocoder'].split('.')[-1]].wav2spec(temp_dict['wav_fn'])
    processed_input = {
        'item_name': item_name, 'mel': mel,
        'sec': len(wav) / hparams['audio_sample_rate'], 'len': mel.shape[0]
    }
    processed_input = {**temp_dict, **processed_input} # merge two dicts

    if binarization_args['with_f0']:
        get_pitch(wav, mel)
    if binarization_args['with_hubert']:
        
        hubert_encoded = processed_input['hubert'] = encoder.encode(temp_dict['wav_fn'])
        if binarization_args['with_align']:
            get_align(temp_dict, mel, hubert_encoded)

    return processed_input
def getitem(item):
    max_frames = hparams['max_frames']
    spec = torch.Tensor(item['mel'])[:max_frames]
    energy = (spec.exp() ** 2).sum(-1).sqrt()
    mel2ph = torch.LongTensor(item['mel2ph'])[:max_frames] if 'mel2ph' in item else None
    f0, uv = norm_interp_f0(item["f0"][:max_frames], hparams)
    #phone = torch.LongTensor(item['phone'][:hparams['max_input_tokens']])
    hubert=torch.Tensor(item['hubert'][:hparams['max_input_tokens']])
    pitch = torch.LongTensor(item.get("pitch"))[:max_frames]
    # print(item.keys(), item['mel'].shape, spec.shape)
    sample = {
        "item_name": item['item_name'],
        # "text": item['txt'],
        # "txt_token": phone,
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
def processed_input2batch(samples):
    '''
        Args:
            samples: one batch of processed_input
        NOTE:
            the batch size is controlled by hparams['max_sentences']
    '''
    if len(samples) == 0:
        return {}
    # id = torch.LongTensor([s['id'] for s in samples])
    item_names = [s['item_name'] for s in samples]
    hubert = utils.collate_2d([s['hubert'] for s in samples], 0.0)
    f0 = utils.collate_1d([s['f0'] for s in samples], 0.0)
    pitch = utils.collate_1d([s['pitch'] for s in samples])
    uv = utils.collate_1d([s['uv'] for s in samples])
    energy = utils.collate_1d([s['energy'] for s in samples], 0.0)
    mel2ph = utils.collate_1d([s['mel2ph'] for s in samples], 0.0) \
        if samples[0]['mel2ph'] is not None else None
    mels = utils.collate_2d([s['mel'] for s in samples], 0.0)
    #hubert_lengths = torch.LongTensor([s['hubert'].shape[0] for s in samples])
    mel_lengths = torch.LongTensor([s['mel'].shape[0] for s in samples])

    batch = {
        # 'id': id,
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

def test_step(sample,key,use_pe=True):
    # print(sample)
    spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_ids')
    hubert = sample['hubert']
    mel2ph, uv, f0 = None, None, None
    ref_mels = sample["mels"]
    mel2ph = sample['mel2ph']
    # sample['f0'][:,3300:3500]=sample['f0'][:,3300]
    sample['f0'] = sample['f0']+(key/12)
    f0 = sample['f0']
    uv = sample['uv']
    outputs = model(
            hubert.cuda(), spk_embed=spk_embed, mel2ph=mel2ph.cuda(), f0=f0.cuda(), uv=uv.cuda(), ref_mels=ref_mels.cuda(), infer=True)
    sample['outputs'] = model.out2mel(outputs['mel_out'])
    sample['mel2ph_pred'] = outputs['mel2ph']
    sample['f0'] = denorm_f0(sample['f0'], sample['uv'], hparams)
    if use_pe:
        sample['f0_pred'] = pe(outputs['mel_out'])['f0_denorm_pred'].detach()#pe(ref_mels.cuda())['f0_denorm_pred'].detach()#
    else:
        sample['f0_pred'] = outputs.get('f0_denorm')
    return after_infer(sample)

def after_infer(prediction):
    vocoder: BaseVocoder = get_vocoder_cls(hparams)()
    for k, v in prediction.items():
        if type(v) is torch.Tensor:
            prediction[k] = v.cpu().numpy()

    item_name = prediction.get('item_name')
    #text = prediction.get('text').replace(":", "%3A")[:80]

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

    f0_gt = prediction.get("f0")
    f0_pred = prediction.get("f0_pred")
    if f0_pred is not None:
        f0_gt = f0_gt[mel_gt_mask]
    if len(f0_pred) > len(mel_pred_mask):
        f0_pred = f0_pred[:len(mel_pred_mask)]
    f0_pred = f0_pred[mel_pred_mask]
    wav_pred = vocoder.spec2wav(mel_pred, f0=f0_pred)
    return f0_gt,f0_pred,wav_pred