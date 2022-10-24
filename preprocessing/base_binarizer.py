import os
from webbrowser import get
os.environ["OMP_NUM_THREADS"] = "1"
import yaml
from utils.multiprocess_utils import chunked_multiprocess_run
import random
import json
# from resemblyzer import VoiceEncoder
from tqdm import tqdm
from preprocessing.data_gen_utils import get_mel2ph, get_pitch_parselmouth, build_phone_encoder,get_pitch_crepe
from utils.hparams import set_hparams, hparams
import numpy as np
from utils.indexed_datasets import IndexedDatasetBuilder


class BinarizationError(Exception):
    pass

BASE_ITEM_ATTRIBUTES = ['txt', 'ph', 'wav_fn', 'tg_fn', 'spk_id']

class BaseBinarizer:
    '''
        Base class for data processing.
        1. *process* and *process_data_split*:
            process entire data, generate the train-test split (support parallel processing);
        2. *process_item*:
            process singe piece of data;
        3. *get_pitch*:
            infer the pitch using some algorithm;
        4. *get_align*:
            get the alignment using 'mel2ph' format (see https://arxiv.org/abs/1905.09263).
        5. phoneme encoder, voice encoder, etc.

        Subclasses should define:
        1. *load_metadata*:
            how to read multiple datasets from files;
        2. *train_item_names*, *valid_item_names*, *test_item_names*:
            how to split the dataset;
        3. load_ph_set:
            the phoneme set.
    '''
    def __init__(self, item_attributes=BASE_ITEM_ATTRIBUTES):
        self.binarization_args = hparams['binarization_args']
        #self.pre_align_args = hparams['pre_align_args']
        
        self.items = {}
        # every item in self.items has some attributes
        self.item_attributes = item_attributes

        self.load_meta_data()
        # check program correctness 检查itemdict的key只能在给定的列表中取值
        assert all([attr in self.item_attributes for attr in list(self.items.values())[0].keys()])
        self.item_names = sorted(list(self.items.keys()))
        
        if self.binarization_args['shuffle']:
            random.seed(1234)
            random.shuffle(self.item_names)
        
        # set default get_pitch algorithm
        if hparams['use_crepe']:
            self.get_pitch_algorithm = get_pitch_crepe
        else:
            self.get_pitch_algorithm = get_pitch_parselmouth

    def load_meta_data(self):
        raise NotImplementedError

    @property
    def train_item_names(self):
        raise NotImplementedError
        
    @property
    def valid_item_names(self):
        raise NotImplementedError

    @property
    def test_item_names(self):
        raise NotImplementedError

    def build_spk_map(self):
        spk_map = set()
        for item_name in self.item_names:
            spk_name = self.items[item_name]['spk_id']
            spk_map.add(spk_name)
        spk_map = {x: i for i, x in enumerate(sorted(list(spk_map)))}
        assert len(spk_map) == 0 or len(spk_map) <= hparams['num_spk'], len(spk_map)
        return spk_map

    def item_name2spk_id(self, item_name):
        return self.spk_map[self.items[item_name]['spk_id']]

    def _phone_encoder(self):
        '''
        use hubert encoder
        '''
        raise NotImplementedError
        '''
            create 'phone_set.json' file if it doesn't exist
        '''
        ph_set_fn = f"{hparams['binary_data_dir']}/phone_set.json"
        ph_set = []
        if hparams['reset_phone_dict'] or not os.path.exists(ph_set_fn):
            self.load_ph_set(ph_set)
            ph_set = sorted(set(ph_set))
            json.dump(ph_set, open(ph_set_fn, 'w', encoding='utf-8'))
            print("| Build phone set: ", ph_set)
        else:
            ph_set = json.load(open(ph_set_fn, 'r', encoding='utf-8'))
            print("| Load phone set: ", ph_set)
        return build_phone_encoder(hparams['binary_data_dir'])
    

    def load_ph_set(self, ph_set):
        raise NotImplementedError

    def meta_data_iterator(self, prefix):
        if prefix == 'valid':
            item_names = self.valid_item_names
        elif prefix == 'test':
            item_names = self.test_item_names
        else:
            item_names = self.train_item_names
        for item_name in item_names:
            meta_data = self.items[item_name]
            yield item_name, meta_data

    def process(self):
        os.makedirs(hparams['binary_data_dir'], exist_ok=True)
        self.spk_map = self.build_spk_map()
        print("| spk_map: ", self.spk_map)
        spk_map_fn = f"{hparams['binary_data_dir']}/spk_map.json"
        json.dump(self.spk_map, open(spk_map_fn, 'w', encoding='utf-8'))

        self.phone_encoder =self._phone_encoder()
        self.process_data_split('valid')
        self.process_data_split('test')
        self.process_data_split('train')

    def process_data_split(self, prefix):
        data_dir = hparams['binary_data_dir']
        args = []
        builder = IndexedDatasetBuilder(f'{data_dir}/{prefix}')
        lengths = []
        f0s = []
        total_sec = 0
        # if self.binarization_args['with_spk_embed']:
        #     voice_encoder = VoiceEncoder().cuda()

        for item_name, meta_data in self.meta_data_iterator(prefix):
            args.append([item_name, meta_data, self.binarization_args])
        spec_min=[]
        spec_max=[]
        # code for single cpu processing
        for i in tqdm(reversed(range(len(args))), total=len(args)):
            a = args[i]
            item = self.process_item(*a)
            if item is None:
                continue
            spec_min.append(item['spec_min'])
            spec_max.append(item['spec_max'])
            # item['spk_embe'] = voice_encoder.embed_utterance(item['wav']) \
            #     if self.binardization_args['with_spk_embed'] else None
            if not self.binarization_args['with_wav'] and 'wav' in item:
                if hparams['debug']:
                    print("del wav")
                del item['wav']
            if(hparams['debug']):
                print(item)
            builder.add_item(item)
            lengths.append(item['len'])
            total_sec += item['sec']
            # if item.get('f0') is not None:
            #     f0s.append(item['f0'])
        if prefix=='train':
            spec_max=np.max(spec_max,0)
            spec_min=np.min(spec_min,0)
            print(spec_max.shape)
            with open(hparams['config_path'], encoding='utf-8') as f:
                _hparams=yaml.safe_load(f)
                _hparams['spec_max']=spec_max.tolist()
                _hparams['spec_min']=spec_min.tolist()
            with open(hparams['config_path'], 'w', encoding='utf-8') as f:
                yaml.safe_dump(_hparams,f)
        builder.finalize()
        np.save(f'{data_dir}/{prefix}_lengths.npy', lengths)
        if len(f0s) > 0:
            f0s = np.concatenate(f0s, 0)
            f0s = f0s[f0s != 0]
            np.save(f'{data_dir}/{prefix}_f0s_mean_std.npy', [np.mean(f0s).item(), np.std(f0s).item()])
        print(f"| {prefix} total duration: {total_sec:.3f}s")

    def process_item(self, item_name, meta_data, binarization_args):
        from preprocessing.process_pipeline import File2Batch
        return File2Batch.temporary_dict2processed_input(item_name, meta_data, self.phone_encoder, binarization_args)

    def get_align(self, meta_data, mel, phone_encoded, res):
        raise NotImplementedError

    def get_align_from_textgrid(self, meta_data, mel, phone_encoded, res):
        '''
            NOTE: this part of script is *isolated* from other scripts, which means
                  it may not be compatible with the current version.
        '''
        return
        tg_fn, ph = meta_data['tg_fn'], meta_data['ph']
        if tg_fn is not None and os.path.exists(tg_fn):
            mel2ph, dur = get_mel2ph(tg_fn, ph, mel, hparams)
        else:
            raise BinarizationError(f"Align not found")
        if mel2ph.max() - 1 >= len(phone_encoded):
            raise BinarizationError(
                f"Align does not match: mel2ph.max() - 1: {mel2ph.max() - 1}, len(phone_encoded): {len(phone_encoded)}")
        res['mel2ph'] = mel2ph
        res['dur'] = dur

    def get_f0cwt(self, f0, res):
        '''
            NOTE: this part of script is *isolated* from other scripts, which means
                  it may not be compatible with the current version.
        '''
        return
        from utils.cwt import get_cont_lf0, get_lf0_cwt
        uv, cont_lf0_lpf = get_cont_lf0(f0)
        logf0s_mean_org, logf0s_std_org = np.mean(cont_lf0_lpf), np.std(cont_lf0_lpf)
        cont_lf0_lpf_norm = (cont_lf0_lpf - logf0s_mean_org) / logf0s_std_org
        Wavelet_lf0, scales = get_lf0_cwt(cont_lf0_lpf_norm)
        if np.any(np.isnan(Wavelet_lf0)):
            raise BinarizationError("NaN CWT")
        res['cwt_spec'] = Wavelet_lf0
        res['cwt_scales'] = scales
        res['f0_mean'] = logf0s_mean_org
        res['f0_std'] = logf0s_std_org


if __name__ == "__main__":
    set_hparams()
    BaseBinarizer().process()
