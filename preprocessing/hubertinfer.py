import os.path
from pathlib import Path

import numpy as np
import torch

from network.hubert.hubert_model import hubert_soft, get_units
from utils.hparams import hparams


class Hubertencoder():
    def __init__(self, pt_path='checkpoints/hubert/hubert_soft.pt'):
        pt_path = list(Path(pt_path).parent.rglob('*.pt'))[0]
        if 'hubert_gpu' in hparams.keys():
            self.use_gpu = hparams['hubert_gpu']
        else:
            self.use_gpu = True
        self.dev = torch.device("cuda" if self.use_gpu and torch.cuda.is_available() else "cpu")
        self.hbt_model = hubert_soft(str(pt_path))

    def encode(self, wav_path):
        npy_path = Path(wav_path).with_suffix('.npy')
        if os.path.exists(npy_path):
            units = np.load(str(npy_path))
        else:
            # sourceonnx = {"source": np.expand_dims(np.expand_dims(wav16, 0), 0)}
            # unitsonnx = np.array(self.hubertsession.run(['embed'], sourceonnx)[0][0])
            units = get_units(self.hbt_model, wav_path, self.dev).cpu().numpy()[0]
            # print(units.shape)
        return units  # [T,256]
