import io
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile

from infer_tools import infer_tool
from infer_tools import slicer
from infer_tools.infer_tool import Svc
from utils.hparams import hparams

chunks_dict = infer_tool.read_temp("./infer_tools/new_chunks_temp.json")


def run_clip(svc_model, key, acc, use_pe, use_crepe, thre, use_gt_mel, add_noise_step, project_name='', f_name=None,
             file_path=None, out_path=None, slice_db=-40,**kwargs):
    print(f'code version:2022-12-04')
    use_pe = use_pe if hparams['audio_sample_rate'] == 24000 else False
    if file_path is None:
        raw_audio_path = f"./raw/{f_name}"
        clean_name = f_name[:-4]
    else:
        raw_audio_path = file_path
        clean_name = str(Path(file_path).name)[:-4]
    infer_tool.format_wav(raw_audio_path)
    wav_path = Path(raw_audio_path).with_suffix('.wav')
    global chunks_dict
    audio, sr = librosa.load(wav_path, mono=True,sr=None)
    wav_hash = infer_tool.get_md5(audio)
    if wav_hash in chunks_dict.keys():
        print("load chunks from temp")
        chunks = chunks_dict[wav_hash]["chunks"]
    else:
        chunks = slicer.cut(wav_path, db_thresh=slice_db)
    chunks_dict[wav_hash] = {"chunks": chunks, "time": int(time.time())}
    infer_tool.write_temp("./infer_tools/new_chunks_temp.json", chunks_dict)
    audio_data, audio_sr = slicer.chunks2audio(wav_path, chunks)

    count = 0
    f0_tst = []
    f0_pred = []
    audio = []
    for (slice_tag, data) in audio_data:
        print(f'#=====segment start, {round(len(data) / audio_sr, 3)}s======')
        length = int(np.ceil(len(data) / audio_sr * hparams['audio_sample_rate']))
        raw_path = io.BytesIO()
        soundfile.write(raw_path, data, audio_sr, format="wav")
        if hparams['debug']:
            print(np.mean(data), np.var(data))
        raw_path.seek(0)
        if slice_tag:
            print('jump empty segment')
            _f0_tst, _f0_pred, _audio = (
                np.zeros(int(np.ceil(length / hparams['hop_size']))), np.zeros(int(np.ceil(length / hparams['hop_size']))),
                np.zeros(length))
        else:
            _f0_tst, _f0_pred, _audio = svc_model.infer(raw_path, key=key, acc=acc, use_pe=use_pe, use_crepe=use_crepe,
                                                        thre=thre, use_gt_mel=use_gt_mel, add_noise_step=add_noise_step)
        fix_audio = np.zeros(length)
        fix_audio[:] = np.mean(_audio)
        fix_audio[:len(_audio)] = _audio[0 if len(_audio)<len(fix_audio) else len(_audio)-len(fix_audio):]
        f0_tst.extend(_f0_tst)
        f0_pred.extend(_f0_pred)
        audio.extend(list(fix_audio))
        count += 1
    if out_path is None:
        out_path = f'./results/{clean_name}_{key}key_{project_name}_{hparams["residual_channels"]}_{hparams["residual_layers"]}_{int(step / 1000)}k_{accelerate}x.{kwargs["format"]}'
    soundfile.write(out_path, audio, hparams["audio_sample_rate"], 'PCM_16',format=out_path.split('.')[-1])
    return np.array(f0_tst), np.array(f0_pred), audio


if __name__ == '__main__':
    # 工程文件夹名，训练时用的那个
    project_name = "yilanqiu"
    model_path = f'./checkpoints/{project_name}/model_ckpt_steps_246000.ckpt'
    config_path = f'./checkpoints/{project_name}/config.yaml'

    # 支持多个wav/ogg文件，放在raw文件夹下，带扩展名
    file_names = ["青花瓷.wav"]
    trans = [0]  # 音高调整，支持正负（半音），数量与上一行对应，不足的自动按第一个移调参数补齐
    # 加速倍数
    accelerate = 20
    hubert_gpu = True
    format='flac'
    step = int(model_path.split("_")[-1].split(".")[0])

    # 下面不动
    infer_tool.mkdir(["./raw", "./results"])
    infer_tool.fill_a_to_b(trans, file_names)

    model = Svc(project_name, config_path, hubert_gpu, model_path)
    for f_name, tran in zip(file_names, trans):
        if "." not in f_name:
            f_name += ".wav"
        run_clip(model, key=tran, acc=accelerate, use_crepe=True, thre=0.05, use_pe=True, use_gt_mel=False,
                 add_noise_step=500, f_name=f_name, project_name=project_name, format=format)
