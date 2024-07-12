import torch
import numpy as np
import torch.nn as nn
from pytorch_fid.fid_score import calculate_frechet_distance
from collections import OrderedDict as OD

from src.utils.utils import convert_bbox
from src.utils.fid_model import LayoutNet

class FID_score(nn.Module):

    def __init__(self, dataset='', data_path='', calc_every_n=50):
        super().__init__()
        self.dataset = dataset
        self.device = torch.device('cpu')
        self.calc_every_n = calc_every_n
        
        # Load pre-calculated mean (mu) and variance (sig) of val/test feature vectors for FID calculation
        musig = torch.load(f'./pretrained/FIDNet_musig_val_{dataset.lower()}.pt', map_location='cpu')
        self.mu, self.sig = musig[0].detach().numpy(), musig[1:].detach().numpy()
        musig_test = torch.load(f'./pretrained/FIDNet_musig_test_{self.dataset.lower()}.pt', map_location='cpu')
        self.mu_test, self.sig_test = musig_test[0].detach().numpy(), musig_test[1:].detach().numpy()
        
        num_classes = 5 if dataset == 'PubLayNet' else 25
        self.fid_model = LayoutNet(num_classes, 20)
        # load pre-trained LayoutNet
        state_dict = torch.load(f'./pretrained/fid_{dataset.lower()}.pth.tar', map_location='cpu')
        # remove "module" prefix if necessary
        state = OD([(key.split("module.")[-1], state_dict[key]) for key in state_dict])

        self.fid_model.load_state_dict(state)
        self.fid_model.requires_grad_(False)
        self.fid_model.eval()

        for param in self.fid_model.parameters():
            param.requires_grad = False

        self.sanity_check()

    
    def sanity_check(self):
        fid_val_test = calculate_frechet_distance(self.mu_test, self.sig_test, self.mu, self.sig)
        print(f"FID sanity check for {self.dataset}: {fid_val_test:.4f}")

    def calc_FID(self, data, format='xywh', test=False):
        '''
        calculates the FID score between the data and the validation or test set
        
        data (dict): containing following elements
        - bbox (N, S, 4): bounding box between with values between 0 and 1
        - label (N, S, 1): category labels (usually integer values indicating the class, with 0 corresponding to padding) 
        - pad_mask (N, S, 1): binary mask, where 0 corresponds to padding
        N = number of layouts
        S = sequence length
        format (str): indicates data format
        test (bool): indicates whether FID should be calculated using the test data
        '''

        try:
            with torch.no_grad():
                if format != 'ltrb':
                    ltrb_bbox = convert_bbox(data['bbox'], f'{format}->ltrb') * data['pad_mask'][..., None]
                else:
                    ltrb_bbox = data['bbox']
                feats_fake = self.fid_model.extract_features(ltrb_bbox, data['label'], (~data['pad_mask']))
                fake_mu, fake_sig = np.mean(feats_fake.cpu().numpy(), axis=0), np.cov(feats_fake.cpu().numpy(), rowvar=False)
            if test:
                mu, sig = self.mu_test, self.sig_test
            else:
                mu, sig = self.mu, self.sig            
            return calculate_frechet_distance(fake_mu, fake_sig, mu, sig)
        except:
            print('Failed to calculate FID.')
            return -1