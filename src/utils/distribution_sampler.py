import torch
import torch.nn as nn

class DistributionSampler(nn.Module):
    '''
    Implementation of various initial distributions.
    '''

    def __init__(self,
        distribution='gaussian',
        sample_padding = False, 
        out_dim = 4,
        sigma = 0.1,
        frame_width = 0.5,
        ):
        super().__init__()
        self.sample_padding = sample_padding
        self.out_dim = out_dim
        self.distribution = distribution
        self.conditional = False
        
        if distribution == 'gaussian':
            self.sampler = self.sample_gaussian
        elif distribution == 'uniform':
            self.sampler = self.sample_uniform
        elif distribution == 'gauss_uniform':
            self.sampler = self.sample_gauss_uniform
            self.sigma = sigma
        elif distribution == '4-gaussians':
            self.sampler = self.sample_4gaussians
            self.sigma = sigma
        elif distribution == 'cond_6-gaussians':
            self.sampler = self.sample_cond_6gaussians
            self.sigma = sigma
            self.conditional = True
        elif distribution == 'frame':
            self.sampler = self.sample_frame
            self.d = frame_width
        else:
            print("Invalid Sampler")

    def sample(self, batch):
        '''
        sample batch according to initialized sampler
        '''
        self.B, self.S = batch['type'].shape
        self.device = batch['type'].device

        if self.sample_padding:
            if not self.conditional:
                x0 = self.sampler((self.B, self.S))
            else:
                x0 = self.sampler((self.B, self.S), batch['type'])
        else:
            x0 = torch.zeros((self.B, self.S, self.out_dim), device=self.device)
            for i in range(self.B):
                L = batch['length'][i]
                if not self.conditional:
                    x0[i, :L] = self.sampler((L,))
                else:
                    x0[i, :L] = self.sampler((L,), batch['type'][i, :L])                
        return x0

    def preprocess(self, data, reverse=False):
        '''
        preprocess data according to sampler (e.g. if samples are gaussian move range of data to [-1, 1])
        assumes data is already normalized to [0, 1] 
        '''
        if self.distribution in ['gaussian', 'gmm', 'uniform', 'gauss_uniform']:
            return 2*data-1 if not reverse else (data + 1)/2
        else:
            return data

    def sample_gaussian(self, size):
        return torch.randn((*size, self.out_dim), device=self.device)
    
    def sample_uniform(self, size):
        return 2*torch.rand((*size, self.out_dim), device=self.device)-1
    
    def sample_gauss_uniform(self, size):
        geom = torch.randn((*size, 4), device=self.device)
        attr = 2*torch.rand((*size, self.out_dim-4), device=self.device)-1
        return torch.cat([geom, attr], dim=-1)
    
    def sample_4gaussians(self, size):
        m = torch.randint(0, 2, (*size, 2), device=self.device)
        d = self.sigma*torch.randn((*size, 4), device=self.device)
        x = torch.cat([m, m], dim=-1) + d
        if self.out_dim > 4:
            x2 = torch.randn((*size, self.out_dim-4), device=self.device)
            x = torch.cat([x, x2], dim=-1)
        return x

    def sample_cond_6gaussians(self, size, type):
        m = torch.stack([torch.cos(2*torch.pi/6*type)+0.5, torch.sin(2*torch.pi/6*type)+0.5], dim=-1)
        d = self.sigma*torch.randn((*size, 4))
        x = torch.cat([m, m], dim=-1) + d.to(type.device)
        if self.out_dim > 4:
            x2 = torch.randn((*size, self.out_dim-4), device=self.device)
            x = torch.cat([x, x2], dim=-1)
        return x

    def sample_frame(self, size):
        x = torch.zeros((*size, 4), device=sefl.device)
        if len(size) == 1:
            for i, params in enumerate(torch.rand(size[-1], 6)):
                x[i, :2] = self.get_frame_coord(*params[:3])
                x[i, 2:] = self.get_frame_coord(*params[3:])
        if len(size) == 2:
            for i in range(size[0]):
                for j, params in enumerate(torch.rand(size[-1], 6)):
                    x[i, j, :2] = self.get_frame_coord(*params[:3])
                    x[i, j, 2:] = self.get_frame_coord(*params[3:])
        if self.out_dim > 4:
            x2 = torch.randn((*size, self.out_dim-4), device=self.device)
            x = torch.cat([x, x2], dim=-1)
        return x

    def get_frame_coord(self, side, t, dt):
        dt = dt*self.d
        if int(4*side) == 0:
            x = 0-dt
            y = t*(1+self.d)
        elif int(4*side) == 1:
            x = t*(1+self.d)-self.d
            y = 0-dt
        elif int(4*side) == 2:
            x = 1+dt
            y = t*(1+self.d)-self.d
        else:
            x = t*(1+self.d)
            y = 1+dt
        return torch.stack([x, y])
