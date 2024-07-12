import numpy as np
import torch
import h5pickle as h5py
from torch.utils.data import Dataset, default_collate

labels = [
    'text',
    'title',
    'list',
    'table',
    'figure',
]

def collate_fn(batch, max_len=None, format='xywh'):

    total_elems = [len(example["type"]) for example in batch]
    max_len = max(total_elems) if max_len is None else max_len
    B = len(batch)

    for key in ['mask', 'length', 'type', 'bbox']:
        dtype = torch.float32
        if key == 'type':
            dtype = torch.int
            size = (max_len,)
        elif key == 'bbox':
            size = (max_len, 4)
        elif key == 'mask':
            dtype = torch.bool
            size = (max_len, 1)
        else:
            dtype = torch.int
            size = (max_len,)

        kwargs = {"fill_value": 0, "dtype": dtype, "size": size}
        for i in range(B):
            dummy_array = torch.full(**kwargs)
            if key in ['length', 'mask']:
                L = batch[i]['length'].squeeze().item()
                if key == 'length':
                    dummy_array = L if max_len is None else min(L, max_len)
                else:
                    dummy_array[:L] = True
            elif key == 'bbox':
                dummy_array[: total_elems[i]] = batch[i][key][:max_len]
                if format == 'xywh':
                    dummy_array[:,0] += dummy_array[:,2]/2
                    dummy_array[:,1] += dummy_array[:,3]/2
                elif format == 'ltrb':
                    dummy_array[:,2] += dummy_array[:,0]
                    dummy_array[:,3] += dummy_array[:,1]
            else:
                dummy_array[: total_elems[i]] = batch[i][key][:max_len]
            batch[i][key] = dummy_array

    return default_collate(batch)


class PubLayNet(Dataset):

    def __init__(self, split='train', data_path='./LayoutDM', num_cat=6, lex_order=False, permute_elements=False, inoue_split=False):
        super().__init__()
        if split == 'train':
            if inoue_split:
                self.data = h5py.File(f'{data_path}/publaynet_train_inoue.h5')
            else:
                self.data = h5py.File(f'{data_path}/publaynet_train.h5' if not lex_order else f'{data_path}/ldm_lex_publaynet_train.h5')
        elif split == 'validation':
            if inoue_split:
                self.data = h5py.File(f'{data_path}/publaynet_val_inoue.h5')
            else:
                self.data = h5py.File(f'{data_path}/publaynet_val.h5' if not lex_order else f'{data_path}/ldm_lex_publaynet_val.h5')
        elif split == 'test':
            if inoue_split:
                self.data = h5py.File(f'{data_path}/publaynet_test_inoue.h5')
            else:
                self.data = h5py.File(f'{data_path}/publaynet_test.h5' if not lex_order else f'{data_path}/ldm_lex_publaynet_test.h5')
        self.keys = list(self.data.keys())
        self.permute_elements = permute_elements

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]
        sample = self.data[key]
        sample_dict = {}
        for feature in sample.keys():
            sample_dict[feature] = torch.from_numpy(np.array(sample[feature])) 

        if 'categories' in sample_dict:
            sample_dict['type'] = sample_dict['categories']
            del sample_dict['categories']

        if self.permute_elements:
            randperm = torch.randperm(sample_dict['length'])
            sample_dict['type'] = sample_dict['type'][randperm]
            sample_dict['bbox'] = sample_dict['bbox'][randperm]
        
        return sample_dict
    
