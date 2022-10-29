import logging

import soundfile

from infer import infer_tool
from infer import merge
from infer.infer_tool import Svc

logging.getLogger('numba').setLevel(logging.WARNING)

# 工程文件夹名，训练时用的那个
project_name = "yilanqiu"
model_path = f'./checkpoints/{project_name}/model_ckpt_steps_44000.ckpt'

# 支持多个wav文件，放在raw文件夹下
clean_names = ["十年"]
trans = [-6]  # 音高调整，支持正负（半音）
# 加速倍数
accelerate = 50

# 下面不动
infer_tool.mkdir(["./raw", "./results"])

input_wav_path = "./infer/wav_temp/input"
out_wav_path = "./infer/wav_temp/output"
cut_time = 30

svc_model = Svc(project_name, model_path)
infer_tool.fill_a_to_b(trans, clean_names)
infer_tool.mkdir(["./infer/wav_temp", input_wav_path, out_wav_path])

# 清除缓存文件
infer_tool.del_temp_wav(input_wav_path)
for clean_name, tran in zip(clean_names, trans):
    raw_audio_path = f"./raw/{clean_name}.wav"
    infer_tool.del_temp_wav("./infer/wav_temp")
    out_audio_name = clean_name
    infer_tool.cut_wav(raw_audio_path, out_audio_name, input_wav_path, cut_time)

    count = 0
    file_list = infer_tool.get_end_file(input_wav_path, "wav")
    for file_name in file_list:
        file_name = file_name.split("/")[-1]
        raw_path = f"{input_wav_path}/{file_name}"
        out_path = f"{out_wav_path}/{file_name}"

        audio = svc_model.infer(raw_path, key=tran, acc=accelerate, use_pe=True, use_gt_mel=False, add_noise_step=500)
        soundfile.write(out_path, audio, 24000, 'PCM_16')

        count += 1
    merge.run(out_audio_name, f"_{tran}key_{project_name}")
    # 清除缓存文件
    infer_tool.del_temp_wav(out_wav_path)
