import os

from pydub import AudioSegment

out_wav_path = "./infer/wav_temp/output"


def wav_combine(*args):
    n = args[0][0]  # 需要拼接的wav个数
    i = 1
    sounds = []
    while i <= n:
        sounds.append(AudioSegment.from_wav(args[0][i]))
        i += 1
    playlist = AudioSegment.empty()
    for sound in sounds:
        playlist += sound
    playlist.export(args[0][n + 1], format="wav")


def run(out_name, end):
    file_list = os.listdir(out_wav_path)
    in_files = [len(file_list)]
    for i in range(0, len(file_list)):
        in_files.append(f"{out_wav_path}/{out_name}-%s.wav" % str(i).zfill(2))
    out_path = f'./results/{out_name}{end}.wav'
    in_files.append(out_path)
    wav_combine(in_files)
    print("out diff-svc success")
    # infer_tool.del_temp_wav("./wav_temp")
