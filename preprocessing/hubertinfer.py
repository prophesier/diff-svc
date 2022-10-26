import onnxruntime
import librosa
import numpy as np
from utils.hparams import hparams

class Hubertencoder():
    def __init__(self,onnxpath='checkpoints/hubert/hubert.onnx'):
        if 'hubert_gpu' in hparams.keys():
            self.use_gpu=hparams['hubert_gpu']
        else:
            self.use_gpu=True
        if self.use_gpu:
            self.hubertsession=onnxruntime.InferenceSession(onnxpath,providers=['CUDAExecutionProvider'])
        else:
            self.hubertsession=onnxruntime.InferenceSession(onnxpath,providers=['CPUExecutionProvider'])

    def encode(self,wavpath):
        wav, sr = librosa.load(wavpath,sr=None)
        assert(sr>=16000)
        if len(wav.shape) > 1:
            wav = librosa.to_mono(wav) 
        if sr!=16000:  
            wav16 = librosa.resample(wav, sr, 16000)
        else:
            wav16=wav
        sourceonnx = {"source":np.expand_dims(np.expand_dims(wav16,0),0)}
        unitsonnx = np.array(self.hubertsession.run(['embed'], sourceonnx)[0][0])
        return unitsonnx #[T,256]