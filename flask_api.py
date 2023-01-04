import io
import logging

import librosa
import soundfile
from flask import Flask, request, send_file
from flask_cors import CORS

from infer_tools.infer_tool import Svc
from utils.hparams import hparams

app = Flask(__name__)

CORS(app)

logging.getLogger('numba').setLevel(logging.WARNING)


@app.route("/voiceChangeModel", methods=["POST"])
def voice_change_model():
    request_form = request.form
    wave_file = request.files.get("sample", None)
    # 变调信息
    f_pitch_change = float(request_form.get("fPitchChange", 0))
    # DAW所需的采样率
    daw_sample = int(float(request_form.get("sampleRate", 0)))
    speaker_id = int(float(request_form.get("sSpeakId", 0)))
    # http获得wav文件并转换
    input_wav_path = io.BytesIO(wave_file.read())
    # 模型推理
    _f0_tst, _f0_pred, _audio = model.infer(input_wav_path, key=f_pitch_change, acc=accelerate, use_pe=False,
                                            use_crepe=False)
    tar_audio = librosa.resample(_audio, hparams["audio_sample_rate"], daw_sample)
    # 返回音频
    out_wav_path = io.BytesIO()
    soundfile.write(out_wav_path, tar_audio, daw_sample, format="wav")
    out_wav_path.seek(0)
    return send_file(out_wav_path, download_name="temp.wav", as_attachment=True)


if __name__ == '__main__':
    # 工程文件夹名，训练时用的那个
    project_name = "firefox"
    model_path = f'./ckpts/{project_name}/model_ckpt_steps_188000.ckpt'
    config_path = f'./ckpts/{project_name}/config.yaml'

    # 加速倍数
    accelerate = 50
    hubert_gpu = True

    model = Svc(project_name, config_path, hubert_gpu, model_path)

    # 此处与vst插件对应，不建议更改
    app.run(port=6842, host="0.0.0.0", debug=False, threaded=False)
