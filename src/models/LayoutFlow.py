from typing import Any
import torch
import torch.nn as nn
import numpy as np
from torchcfm import ConditionalFlowMatcher
from torchdyn.core import NeuralODE

from src.models.BaseGenModel import BaseGenModel
from src.utils.fid_calculator import FID_score 
from src.utils.analog_bit import AnalogBit


class LayoutFlow(BaseGenModel):
    def __init__(
            self,
            backbone_model,
            sampler,
            optimizer,
            scheduler=None,
            loss_fcn = 'mse', 
            data_path: str = '',
            format = 'xywh',
            sigma = 0.0,
            fid_calc_every_n = 20,
            expname = 'LayoutFlow',
            num_cat = 6,
            time_sampling = 'uniform',
            sample_padding=False,
            inference_steps = 100,
            ode_solver = 'euler',
            cond = 'uncond',
            attr_encoding = 'continuous',
            train_traj = 'linear',
            add_loss = '',
            add_loss_weight=1,
            mask_padding = False,
            cf_guidance=0,
        ):
        self.format = format
        self.init_dist = sampler.distribution
        self.time_sampling = time_sampling
        self.time_weight = float(time_sampling.split('_')[-1]) if 'late_focus_' in time_sampling else 0.3
        self.sample_padding = sample_padding
        self.fid_calc_every_n = fid_calc_every_n
        self.cond = cond
        self.attr_encoding = attr_encoding
        if attr_encoding == 'AnalogBit':
            self.analog_bit = AnalogBit(num_cat)
        if fid_calc_every_n != 0: 
            fid_model = FID_score(dataset=expname.split('_')[0], data_path=data_path, calc_every_n=fid_calc_every_n)
        else:
            fid_model=None
        super().__init__(data_path=data_path, optimizer=optimizer, scheduler=scheduler, expname=expname, fid_model=fid_model)

        self.attr_dim = int(np.ceil(np.log2(num_cat))) if attr_encoding == 'AnalogBit' else 1
        self.num_cat = num_cat
        self.model = backbone_model
        self.inference_steps = inference_steps
        self.ode_solver = ode_solver
        self.sampler = sampler
        self.mask_padding = mask_padding
        self.cf_guidance = cf_guidance

        # Training Parameters
        self.train_traj = train_traj
        self.FM = ConditionalFlowMatcher(sigma=sigma)
        self.loss_fcn = nn.MSELoss() if loss_fcn!='l1' else nn.L1Loss()
        self.add_loss = add_loss
        self.add_loss_weight = add_loss_weight
        self.save_hyperparameters()

        
    def training_step(self, batch, batch_idx):
        # Conditioning Mask
        cond_mask = self.get_cond_mask(batch)
        # Obtain initial sample x_0 and data sample x_1
        x0, x1 = self.get_start_end(batch)
        # Sample timestep
        t = self.sample_t(x0)
        # Calculate intermediate sample x_t based on time and trajectory 
        xt, ut = self.sample_xt(batch, x0, x1, cond_mask, t)

        # Prediction of vector field using backbone model
        vt = self(xt, cond_mask, t.squeeze(-1))

        # Loss calculation
        loss = self.loss_fcn(cond_mask*vt, cond_mask*ut)
        if self.add_loss:
            loss = self.additional_losses(cond_mask, ut, vt, loss) 
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)

        return loss

    def sample_xt(self, batch, x0, x1, cond_mask, t):
        eps = self.FM.sample_noise_like(x0) if self.FM.sigma != 0 else 0
        if self.train_traj == 'sin':
            tpad = t.reshape(-1, *([1] * (x0.dim() - 1)))
            xt = (1 - torch.sin(tpad * torch.pi/2)) * x0 + torch.sin(tpad * torch.pi/2) * x1 + self.FM.sigma * eps
            ut = torch.pi/2 * torch.cos(tpad * torch.pi / 2) * (x1 - x0)
        elif self.train_traj == 'sincos':
            tpad = t.reshape(-1, *([1] * (x0.dim() - 1)))
            xt = torch.cos(tpad * torch.pi/2) * x0 + torch.sin(tpad * torch.pi/2) * x1 + self.FM.sigma * eps
            ut = torch.pi/2 * (torch.cos(tpad * torch.pi / 2) * x1 - torch.sin(tpad * torch.pi / 2) * x0)
        else: #linear
            xt = self.FM.sample_xt(x0, x1, t, eps) 
            ut = self.FM.compute_conditional_flow(x0, x1, t, xt)
        xt = (1-cond_mask) * x1 + cond_mask * xt
        if self.mask_padding:
            xt = batch['mask'] * xt - (~batch['mask']) * torch.ones_like(xt)
            ut *= batch['mask']

        return xt, ut

    def inference(self, batch, full_traj=False, task=None):
        # Sample initial layout x_0 
        x0 = self.sampler.sample(batch)
        # Get conditioning mask
        cond_mask = self.get_cond_mask(batch)
        # Get conditional sample
        if self.attr_encoding == 'AnalogBit':
            conv_type = self.analog_bit.encode(batch['type']) 
        else:
            conv_type = batch['type'].unsqueeze(-1) / (self.num_cat-1)
        ref = torch.cat([batch['bbox'], conv_type], dim=-1)
        cond_x = self.sampler.preprocess(ref)
        cond_x = batch['mask'] * cond_x + (~batch['mask']) * ref
        
        # Create model wrapper for NeuralODE
        if task == 'condinf':
            vector_field = cond_wrapper(self, cond_x, cond_mask, batch=batch)
        else:
            vector_field = torch_wrapper(self, cond_x, cond_mask, cf_guidance=self.cf_guidance)
        
        # Solve NeuralODE
        node = NeuralODE(vector_field, solver=self.ode_solver, sensitivity="adjoint", atol=1e-4, rtol=1e-4)
        if task == 'refinement':
            traj = node.trajectory(cond_x, t_span=torch.linspace(0.97, 1, self.inference_steps))
        else:
            traj = node.trajectory(x0, t_span=torch.linspace(0, 1, self.inference_steps))
        traj = self.sampler.preprocess(traj, reverse=True)
        
        # Post-processing and decoding of obtained trajectory
        if self.attr_encoding == 'AnalogBit':
            cont_cat = self.analog_bit.decode(traj[...,self.geom_dim:])
        else:
            cont_cat = traj[..., -1] * (self.num_cat-1) + 0.5
        cont_cat = (1-cond_mask[:,:,-1]) * batch['type'] + cond_mask[:,:,-1] * cont_cat
        cat = torch.clip(cont_cat.to(torch.int), 0, self.num_cat-1)
        traj = (1-cond_mask[:,:,:self.geom_dim]) * batch['bbox'][None] + cond_mask[:,:,:self.geom_dim] * traj[...,:self.geom_dim]
        self.input_cond = [(1-cond_mask[:,:,:self.geom_dim]) * batch['bbox'], (1-cond_mask[:,:,-1]) * batch['type']]
        
        return (traj, cat, cont_cat) if full_traj else (traj[-1], cat[-1])


class torch_wrapper(torch.nn.Module):
    '''
    Wraps model to torchdyn compatible format.
    forward method defines a single step of the ODE solver.
    '''

    def __init__(self, model, cond_x, cond_mask=None, inverse=False, cf_guidance=0):
        super().__init__()
        self.model = model
        self.cond_x = cond_x
        self.cond_mask = cond_mask
        self.sign = -1 if inverse else 1
        self.cf_guidance = cf_guidance
        if cf_guidance:
            self.uncond_mask = torch.ones_like(self.cond_mask)

    def forward(self, t, x, *args, **kwargs):
        x = (1-self.cond_mask) * self.cond_x + self.cond_mask * x
        if self.sign == -1:
            t = 1 - t
        v = self.model(x, self.cond_mask, t.repeat(x.shape[0]))
        if self.cf_guidance:
            v = (1+self.cf_guidance) * v - self.cf_guidance * self.model(x, self.uncond_mask, t.repeat(x.shape[0]))
        return self.sign * v


class cond_wrapper(torch.nn.Module):
    '''
    Wraps model to torchdyn compatible format.
    This is an alternative conditioning method, that only uses the unconditional masking as described in the Appendix 
    of our paper (Section: Conditioning Analysis).
    '''

    def __init__(self, model, cond_x, cond_mask=None, batch=None):
        super().__init__()
        self.model = model
        self.cond_x = cond_x
        self.cond_mask = cond_mask
        self.batch = batch

    def forward(self, t, x, *args, **kwargs):
        v = self.batch['mask'] * self.model(x, torch.ones_like(self.cond_mask), t.repeat(x.shape[0]))
        cond_dir = (self.cond_x - x)
        new_v = self.cond_mask * v + (1-self.cond_mask) * cond_dir
        return new_v