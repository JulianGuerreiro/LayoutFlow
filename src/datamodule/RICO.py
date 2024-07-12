import numpy as np
import torch
import h5pickle as h5py
from torch.utils.data import Dataset, default_collate


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


class RICO(Dataset):

    def __init__(self, split='train', data_path='./rico', num_cat=26, lex_order=False, permute_elements=False):
        super().__init__()
        
        if split == 'train':
            self.data = h5py.File(f'{data_path}/ldm_rico_train.h5' if not lex_order else f'{data_path}/ldm_lex_rico_train.h5')
        if split == 'validation':
            self.data = h5py.File(f'{data_path}/ldm_rico_val.h5' if not lex_order else f'{data_path}/ldm_lex_rico_val.h5')
        if split == 'test':
            self.data = h5py.File(f'{data_path}/ldm_rico_test.h5' if not lex_order else f'{data_path}/ldm_lex_rico_test.h5')

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
            
        if self.permute_elements:
            randperm = torch.randperm(sample_dict['length'])
            sample_dict['type'] = sample_dict['type'][randperm]
            sample_dict['bbox'] = sample_dict['bbox'][randperm]

        return sample_dict

    
TYPE_2_CAT = {
    'Advertisement': 1,
    'Video': 2,
    'Checkbox': 3,
    'Drawer': 4,
    'Icon': 5,
    'Image': 6,
    'Input': 7,
    'List_Item': 8,
    'Modal': 9,
    'Pager_Indicator': 10,
    'Text': 11,
    'Toolbar': 12,
    'Web_View': 13,
    'Map_View': 14,
    'Text_Button': 15,
    'Background_Image': 16,
    'Slider': 17,
    'Multi_Tab': 18,
    'Radio_Button': 19,
    'Date_Picker': 20,
    'Number_Stepper': 21,
    'Card': 22, 
    'On_Off_Switch': 23,
    'Bottom_Navigation': 24,
    'Button_Bar': 25,
}