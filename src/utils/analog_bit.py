import torch
import numpy as np

class AnalogBit():
    '''
    Implementation of an Analog Bit, mapping categorical variables into continuous Analog Bits.
    '''

    def __init__(self, num_cat, bimod_level=1):
        self.num_cat = num_cat
        self.num_bits = int(np.ceil(np.log2(num_cat)))
        self.b = bimod_level
        self.bit_mask = torch.tensor([1 << k for k in range(self.num_bits)], dtype=torch.long)

    def encode(self, x):
        return torch.bitwise_and(x[:,:,None], self.bit_mask.to(x.device)) / self.bit_mask.to(x.device)
    
    def decode(self, x):
        x = x-0.5
        x = (x + abs(x)).to(torch.bool) * self.bit_mask.to(x.device)
        return torch.sum(x, dim=-1) 