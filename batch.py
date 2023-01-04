import soundfile

from infer_tools import infer_tool
from infer_tools.infer_tool import Svc


def run_clip(svc_model, key, acc, use_pe, use_crepe, thre, use_gt_mel, add_noise_step, project_name='', f_name=None,
             file_path=None, out_path=None):
    raw_audio_path = f_name
    infer_tool.format_wav(raw_audio_path)
    _f0_tst, _f0_pred, _audio = svc_model.infer(raw_audio_path, key=key, acc=acc, singer=True, use_pe=use_pe,
                                                use_crepe=use_crepe,
                                                thre=thre, use_gt_mel=use_gt_mel, add_noise_step=add_noise_step)
    out_path = f'./singer_data/{f_name.split("/")[-1]}'
    soundfile.write(out_path, _audio, 44100, 'PCM_16')


if __name__ == '__main__':
    # 工程文件夹名，训练时用的那个
    project_name = "firefox"
    model_path = f'./ckpts/{project_name}/clean_model_ckpt_steps_100000.ckpt'
    config_path = f'./ckpts/{project_name}/config.yaml'

    # 支持多个wav/ogg文件，放在raw文件夹下，带扩展名
    file_names = infer_tool.get_end_file("./batch", "wav")
    trans = [-6]  # 音高调整，支持正负（半音），数量与上一行对应，不足的自动按第一个移调参数补齐
    # 加速倍数
    accelerate = 50
    hubert_gpu = True
    cut_time = 30

    # 下面不动
    infer_tool.mkdir(["./batch", "./singer_data"])
    infer_tool.fill_a_to_b(trans, file_names)

    model = Svc(project_name, config_path, hubert_gpu, model_path)
    count = 0
    for f_name, tran in zip(file_names, trans):
        print(f_name)
        run_clip(model, key=tran, acc=accelerate, use_crepe=False, thre=0.05, use_pe=False, use_gt_mel=False,
                 add_noise_step=500, f_name=f_name, project_name=project_name)
        count += 1
        print(f"process:{round(count * 100 / len(file_names), 2)}%")
