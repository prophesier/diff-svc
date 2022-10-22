import onnxruntime
import librosa
import numpy as np

class Hubertencoder():
    def __init__(self,onnxpath='checkpoints/hubert/hubert.onnx'):
        self.hubertsession=onnxruntime.InferenceSession(onnxpath,providers=['CPUExecutionProvider','CUDAExecutionProvider'])

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