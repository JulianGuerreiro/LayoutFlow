from typing import Any
import torch
import torch.nn as nn
import numpy as np

from src.models.BaseGenModel import BaseGenModel
from src.utils.fid_calculator import FID_score 
from src.utils.analog_bit import AnalogBit


class LayoutDMx(BaseGenModel):
    def __init__(
        self,
        backbone_model,
        optimizer,
        DM_model,
        sampler,
        scheduler=None,
        loss_fcn = 'mse', 
        data_path: str = '',
        format = 'xywh',
        fid_calc_every_n = 20,
        expname = 'Diffusion',
        num_cat = 6,
        time_sampling = 'uniform',
        sample_padding=False,
        inference_steps = 100,
        cond = 'uncond',
        attr_encoding = 'continuous',
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
        self.DM_model = DM_model
        self.inference_steps = inference_steps
        self.sampler = sampler
        self.mask_padding = mask_padding
        self.cf_guidance = cf_guidance

        # Training Parameters
        self.loss_fcn = nn.MSELoss() if loss_fcn!='l1' else nn.L1Loss()
        self.add_loss = add_loss
        self.add_loss_weight = add_loss_weight
        self.save_hyperparameters()

        
    def training_step(self, batch, batch_idx):
        # Conditioning Mask
        cond_mask = self.get_cond_mask(batch)
        # Obtain initial noise x_0 and data sample x_1
        x0, x1 = self.get_start_end(batch)
        # Time sampling
        t = self.sample_t(x0)
        # Add noise according to time
        xt = self.DM_model.add_noise(x1, x0, (999.99*t).to(torch.int))
        xt = cond_mask * xt + (1-cond_mask) * x1

        # Prediction of initial noise
        x0_hat = self(xt, cond_mask, t.squeeze(-1))

        # Loss calculation
        loss = self.loss_fcn(cond_mask*x0, cond_mask*x0_hat)
        if self.add_loss:
            loss = self.additional_losses(cond_mask, x0, x0_hat, loss)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)

        return loss


    def inference(self, batch, task=None, full_traj=False):        
        # Sample initial noise 
        x0 = self.sampler.sample(batch)
        # Get conditioning mask
        cond_mask = self.get_cond_mask(batch)
        # Get conditioning sample
        if self.attr_encoding == 'AnalogBit':
            conv_type = self.analog_bit.encode(batch['type']) 
        else:
            conv_type = batch['type'].unsqueeze(-1) / (self.num_cat-1)
        gt = torch.cat([batch['bbox'], conv_type], dim=-1)
        cond_x = self.sampler.preprocess(gt)
        cond_x = batch['mask'] * cond_x + (~batch['mask']) * gt
        
        # Sample using denoising model
        self.DM_model.set_timesteps(self.inference_steps)
        input = cond_mask * x0 + (1-cond_mask) * cond_x
        traj = []
        for t in self.DM_model.timesteps:
            timestep = t*torch.ones((x0.shape[0],), device=self.device)/1000
            noisy_residual = self(input, cond_mask, timestep)
            prev_noisy_sample = self.DM_model.step(noisy_residual, t, input).prev_sample
            input = cond_mask * prev_noisy_sample + (1-cond_mask) * cond_x
            traj.append(prev_noisy_sample)
        traj = torch.stack(traj)
        traj = self.sampler.preprocess(traj, reverse=True)
        
        # Post-processing and decoding of obtained trajectory
        if self.attr_encoding == 'AnalogBit':
            cont_cat = self.analog_bit.decode(traj[...,self.geom_dim:])
        else:
            cont_cat = traj[..., -1] * (self.num_cat-1) + 0.5
        cont_cat = (1-cond_mask[:,:,-1]) * batch['type'] + cond_mask[:,:,-1] * cont_cat
        cat = torch.clip(cont_cat.to(torch.int), 0, self.num_cat-1)
        traj = (1-cond_mask[:,:,:self.geom_dim]) * batch['bbox'][None] + cond_mask[:,:,:self.geom_dim] * traj[...,:self.geom_dim]
        
        return (traj, cat, cont_cat) if full_traj else (traj[-1], cat[-1])