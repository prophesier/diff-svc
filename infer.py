import io
from pathlib import Path

import numpy as np
import soundfile

from infer_tools import infer_tool
from infer_tools import slicer
from infer_tools.infer_tool import Svc
from utils.hparams import hparams


def run_clip(svc_model, key, acc, use_pe, use_crepe, thre, use_gt_mel, add_noise_step, project_name='', f_name=None,
             file_path=None, out_path=None):
    if file_path is None:
        raw_audio_path = f"./raw/{f_name}"
        clean_name = f_name[:-4]
    else:
        raw_audio_path = file_path
        clean_name = str(Path(file_path).name)[:-4]
    infer_tool.format_wav(raw_audio_path)
    audio_data, audio_sr = slicer.cut(Path(raw_audio_path).with_suffix('.wav'))

    count = 0
    f0_tst = []
    f0_pred = []
    audio = []
    epsilon = 0.0002
    for data in audio_data:
        print(f'#=====segment start, {round(len(data) / audio_sr, 3)}s======')
        length = int(len(data) / audio_sr * hparams['audio_sample_rate'])
        raw_path = io.BytesIO()
        soundfile.write(raw_path, data, audio_sr, format="wav")
        if hparams['debug']:
            print(np.mean(data), np.var(data))
        raw_path.seek(0)
        if np.var(data) < epsilon:
            print('jump empty segment')
            _f0_tst, _f0_pred, _audio = (
                np.zeros(int(length / hparams['hop_size'])), np.zeros(int(length / hparams['hop_size'])),
                np.zeros(length))
        else:
            _f0_tst, _f0_pred, _audio = svc_model.infer(raw_path, key=key, acc=acc, use_pe=use_pe, use_crepe=use_crepe,
                                                        thre=thre, use_gt_mel=use_gt_mel, add_noise_step=add_noise_step)
        fix_audio = np.zeros(length)
        fix_audio[:] = np.mean(_audio)
        fix_audio[:len(_audio)] = _audio
        f0_tst.extend(_f0_tst)
        f0_pred.extend(_f0_pred)
        audio.extend(list(fix_audio))
        count += 1
    if out_path is None:
        out_path = f'./results/{clean_name}_{key}key_{project_name}_{hparams["residual_channels"]}_{hparams["residual_layers"]}_{int(step / 1000)}k_{accelerate}x.wav'
    soundfile.write(out_path, audio, hparams["audio_sample_rate"], 'PCM_16')
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

    step = int(model_path.split("_")[-1].split(".")[0])

    # 下面不动
    infer_tool.mkdir(["./raw", "./results"])
    infer_tool.fill_a_to_b(trans, file_names)

    model = Svc(project_name, config_path, hubert_gpu, model_path)
    for f_name, tran in zip(file_names, trans):
        run_clip(model, key=tran, acc=accelerate, use_crepe=True, thre=0.05, use_pe=True, use_gt_mel=False,
                 add_noise_step=500, f_name=f_name, project_name=project_name)
