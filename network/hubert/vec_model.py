from pathlib import Path

import librosa
import numpy as np
import torch



def load_model(vec_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("load model(s) from {}".format(vec_path))
    from fairseq import checkpoint_utils
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [vec_path],
        suffix="",
    )
    model = models[0]
    model = model.to(device)
    model.eval()
    return model


def get_vec_units(con_model, audio_path, dev):
    audio, sampling_rate = librosa.load(audio_path)
    if len(audio.shape) > 1:
        audio = librosa.to_mono(audio.transpose(1, 0))
    if sampling_rate != 16000:
        audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=16000)

    feats = torch.from_numpy(audio).float()
    if feats.dim() == 2:  # double channels
        feats = feats.mean(-1)
    assert feats.dim() == 1, feats.dim()
    feats = feats.view(1, -1)
    padding_mask = torch.BoolTensor(feats.shape).fill_(False)
    inputs = {
        "source": feats.to(dev),
        "padding_mask": padding_mask.to(dev),
        "output_layer": 9,  # layer 9
    }
    with torch.no_grad():
        logits = con_model.extract_features(**inputs)
        feats = con_model.final_proj(logits[0])
    return feats


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "../../checkpoints/checkpoint_best_legacy_500.pt"  # checkpoint_best_legacy_500.pt
    vec_model = load_model(model_path)
    # 这个不用改，自动在根目录下所有wav的同文件夹生成其对应的npy
    file_lists = list(Path("../../data/vecfox").rglob('*.wav'))
    nums = len(file_lists)
    count = 0
    for wav_path in file_lists:
        npy_path = wav_path.with_suffix(".npy")
        npy_content = get_vec_units(vec_model, str(wav_path), device).cpu().numpy()[0]
        np.save(str(npy_path), npy_content)
        count += 1
        print(f"hubert process：{round(count * 100 / nums, 2)}%")
