from typing import Any, Optional
import torch
import torch.nn as nn
import lightning.pytorch as pl
from torch import Tensor
import torchvision
import torchvision.transforms.functional as F
import matplotlib
import matplotlib.pyplot as plt
import PIL
import math

from src.utils.metrics import compute_alignment, compute_overlap
from src.utils.visualization import draw_layout
from src.utils.utils import plot_trajectories, convert_bbox

class BaseGenModel(pl.LightningModule):
    def __init__(
            self, 
            optimizer, 
            scheduler=None, 
            data_path='', 
            expname='', 
            fid_model=None,
        ):
        super().__init__()
        self.optimizer_partial = optimizer
        self.scheduler_setting = scheduler
        self.data_path = data_path
        self.dataset = expname.split('_')[0]
        self.fid_model = fid_model
        self.geom_dim = 4 

        self.gen_data = {'bbox': [], 'label': [], 'pad_mask': [], 'seq_lens': [], 'ids': [], 'bbox_cond': [], 'cat_cond': []} 
        self.val_data = {}
        self.idx=[2/256, 39/256, 84/256, 103/256, 180/256, 200/256]
        self.fid_score=0

    def configure_optimizers(self):
        optimizer = self.optimizer_partial(params=self.model.parameters(), betas=(0.9, 0.98))
        if self.scheduler_setting == 'reduce_on_plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
            return [optimizer], [{"scheduler": scheduler, "monitor": "FID_Layout", "frequency": self.fid_model.calc_every_n}]
        else: 
            return optimizer 

    def forward(self, xt, mask_cond, t):
        geom, attr = xt[...,:self.geom_dim], xt[...,self.geom_dim:]
        if self.attr_encoding == 'discrete':
            attr = (self.sampler.preprocess(attr, reverse=True) * (self.num_cat-1) + 0.5).to(int)
            attr = torch.clip(attr, 0, self.num_cat-1)
        return self.model(geom, attr, mask_cond, t)
          
    def get_start_end(self, batch):
        # Encoding for category attribute
        if self.attr_encoding == 'AnalogBit':
            conv_type = self.analog_bit.encode(batch['type']) 
        else:
            conv_type = batch['type'].unsqueeze(-1) / (self.num_cat-1)
        bbox = batch['bbox']
        
        # Conditional Flow Matching
        x0 = self.sampler.sample(batch)
        gt = torch.cat([bbox, conv_type], dim=-1)
        x1 = self.sampler.preprocess(gt)
        x1 = batch['mask'] * x1 + (~batch['mask']) * gt

        return x0, x1
    
    def sample_t(self, x0):
        t = torch.rand(x0.shape[0]).type_as(x0)
        if 'late_focus' in self.time_sampling:
            T_new = int(self.time_weight*len(t))
            t[:T_new] = 0.95 + torch.rand(T_new).type_as(x0)/20 
        return t
    
    def validation_step(self, batch, batch_idx):
        print('Validation_step')
        # if we trained on random masking, perform inference in unconditional setting 
        # and then repeat with conditional setting
        repeat_with_cond = ('random' in self.cond)    
        if repeat_with_cond:
            self.cond = 'uncond'

        # Inference
        geom_pred, cat = self.inference(batch)

        # Validation loss
        geom_gt = batch['bbox']
        loss = self.loss(geom_pred, geom_gt, batch['length'])
        self.log("val_loss", loss, sync_dist=True)

        # Save validation loss
        self.gen_data['bbox'].append(geom_pred)
        self.gen_data['label'].append(cat)    
        pad_mask = torch.zeros(geom_pred.shape[:2], device=self.device, dtype=bool)
        for i, L in enumerate(batch['length']):
            pad_mask[i, :L] = True
        self.gen_data['pad_mask'].append(pad_mask)

        # If random training repeat evaluation in conditional setting
        if repeat_with_cond:
            self.cond = 'cat_cond'
            geom_pred, cat = self.inference(batch, task='cat_cond')
            self.gen_data['bbox_cond'].append(geom_pred)
            self.gen_data['cat_cond'].append(cat)
            self.cond = 'random4'

        if batch_idx == 0:
            self.log("FID_Layout", self.fid_score, on_epoch=True, sync_dist=True)
            # If first batch
            if not 'batch' in self.val_data:
                first_batch = {}
                for key in batch.keys():
                    if key in ["id"]:
                        first_batch[key] = [batch[key][int(i*batch['bbox'].shape[0])] for i in self.idx]
                        continue
                    first_batch[key] = torch.stack([batch[key][int(i*batch['bbox'].shape[0])] for i in self.idx])
                self.val_data["batch"] = first_batch
            self.example_visualization()

    def on_validation_end(self):
        bbox = torch.cat(self.gen_data['bbox'])
        label = torch.cat(self.gen_data['label'])
        pad_mask = torch.cat(self.gen_data['pad_mask'])

        if self.format != 'xywh':
            score_bbox = convert_bbox(bbox, f'{self.format}->xywh') * pad_mask[..., None]
        else:
            score_bbox = bbox
        alignment_score = compute_alignment(score_bbox.cpu(), pad_mask.cpu())
        self.log_dict({"Alignment": alignment_score*100})
        overlap_score = compute_overlap(score_bbox.cpu(), pad_mask.cpu())
        self.log_dict({"Overlap": overlap_score})

        if self.fid_calc_every_n != 0:
            fid_score = self.fid_model.calc_FID({'bbox': bbox, 'label': label, 'pad_mask': pad_mask}, format=self.format)
            self.log_dict({"FID_Layout": fid_score})
            self.fid_score = fid_score
            if 'random' in self.cond:
                bbox_cond = torch.cat(self.gen_data['bbox_cond'])
                cond_label = torch.cat(self.gen_data['cat_cond'])
                cond_fid_score = self.fid_model.calc_FID({'bbox': bbox_cond, 'label': cond_label, 'pad_mask': pad_mask}, format=self.format)
                self.log_dict({"FID_Layout_cond": cond_fid_score})

        for key in self.gen_data:
            self.gen_data[key] = []

    def example_visualization(self):
        batch = self.val_data['batch']
        geom_preds, cats, cont_cats = self.inference(batch, full_traj=True)
        geom_pred, cat = geom_preds[-1], cats[-1]

        geom_preds_var = torch.zeros((4, *geom_preds[0].shape), device=self.device)
        cat_var = torch.zeros((4, *geom_preds[0].shape[:-1]), dtype=torch.int, device=self.device)
        for j in range(4):
            geom_preds_var[j], cat_var[j] = self.inference(batch)

        # Log layout and image
        for i, L in enumerate(batch['length']):
            
            # Visualize prediction and reference image from dataset
            ref_layout = draw_layout(batch['bbox'][i,:L], batch['type'][i,:L], num_colors=self.num_cat, format=self.format)
            pred_layout = draw_layout(geom_pred[i, :L], cat[i,:L], num_colors=self.num_cat, format=self.format)
            self.log_image(
                key=f"[{i}] Layout-Image Comparison",
                images=[ref_layout, pred_layout],
                caption=["Layout Reference (GT)", "Layout Pred"],
            )

            # Visualize layout predictions for different points in time
            images = []
            nlevels = torch.linspace(0, self.inference_steps-1, 10).long()
            for t in nlevels:
                labels = cats[t, i,:L] if self.cond != 'cond_cat' else cat[i, :L]
                images.append(draw_layout(geom_preds[t, i, :L], labels, num_colors=self.num_cat, format=self.format))
            self.log_image(
                key=f"[{i}] Sample Trajectory",
                images = images,
                caption = [f"t={x/100:.2f}" for x in nlevels]
            )

            # Visualize results for various initial samples
            pred_layouts = []
            for j in range(4):
                pred_layouts.append(draw_layout(geom_preds_var[j,i,:L], cat_var[j,i,:L], num_colors=self.num_cat, format=self.format))
            self.log_image(
                key=f"[{i}] Various Samples",
                images=pred_layouts,
                caption=[f"Sample {x+1}" for x in range(4)],
            )

            # Visualize Trajectories over time
            traj_plot_xy = plot_trajectories(geom_preds[:, i, :L, :2].cpu().numpy())
            traj_plot_wh = plot_trajectories(geom_preds[:, i, :L, 2:].cpu().numpy())
            cat_plot = plot_trajectories(cont_cats[:, i, :L].cpu().numpy(), categories=True, num_cat=self.num_cat)
            self.log_image(
                key=f"[{i}] Trajectory", 
                images=[traj_plot_xy, traj_plot_wh, cat_plot],
                caption=[f"({self.format[0]}, {self.format[1]})", f"({self.format[2]}, {self.format[3]})", "Categories"]
            )
            plt.close()
    
    def log_image(self, key, images, caption):
        if isinstance(self.logger, pl.loggers.WandbLogger):
            self.logger.log_image(key=key, images=images, caption=caption)
        else:
            imgs = []
            captions = []
            for tag, img in zip(caption, images):
                if isinstance(img, matplotlib.figure.Figure):
                    img.canvas.draw()
                    img = PIL.Image.frombytes('RGBa', img.canvas.get_width_height(), img.canvas.buffer_rgba())
                imgs.append(F.to_tensor(img))
                captions.append(tag)
            grid_img = torchvision.utils.make_grid(imgs)
            self.logger.experiment.add_image(f'{key}/{"_".join(captions)}', grid_img, self.global_step)

    def log_dict(self, log_dict):
        if isinstance(self.logger, pl.loggers.WandbLogger):
            self.logger.experiment.log(log_dict)
        else:
            for tag, value in log_dict.items():
                self.logger.experiment.add_scalar(f'validation/{tag}', value, self.global_step)

    def get_cond_mask(self, batch):
        if self.cond == 'uncond':
            cond_mask = torch.ones((*batch['bbox'].shape[:2], self.geom_dim+self.attr_dim), dtype=torch.int, device=self.device)
        elif self.cond in ['cat_cond', 'refinement']:
            cond_mask = torch.ones((*batch['bbox'].shape[:2], self.geom_dim+self.attr_dim), dtype=torch.int, device=self.device)
            cond_mask[:,:,self.geom_dim:] = 0
        elif self.cond == 'size_cond':
            cond_mask = torch.ones((*batch['bbox'].shape[:2], self.geom_dim+self.attr_dim), dtype=torch.int, device=self.device)
            cond_mask[:,:,2:] = 0
        elif self.cond == 'elem_compl':
            cond_mask = torch.ones((*batch['bbox'].shape[:2], self.geom_dim+self.attr_dim), dtype=torch.int, device=self.device)
            idx_b, idx_s = [], []
            for i, l in enumerate(batch['length']):
                num_elem = l * 0.2 * torch.rand(1)
                if l > 1:
                    idx = torch.multinomial(torch.arange(l).float(), int(num_elem.item()+1)).tolist()
                    idx_b += [i] * len(idx)
                    idx_s += idx
            idx_b = torch.tensor(idx_b)
            idx_s = torch.tensor(idx_s) 
            cond_mask[idx_b, idx_s] = 0
        elif 'random4' in self.cond:
            cond_mask = torch.ones((*batch['bbox'].shape[:2], self.geom_dim+self.attr_dim), dtype=torch.int, device=self.device)
            div = batch['bbox'].shape[0] // 4
            cond_mask[:div] = 1
            idx_b, idx_s = [], []
            for i, l in enumerate(batch['length'][:div]):
                num_elem = l * 0.2 * torch.rand(1).to(self.device)
                if l > 1:
                    idx = torch.multinomial(torch.arange(l).float(), int(num_elem.item()+1)).tolist()
                    idx_b += [i] * len(idx)
                    idx_s += idx
            idx_b = torch.tensor(idx_b)
            idx_s = torch.tensor(idx_s) 
            cond_mask[idx_b, idx_s] = 0
            cond_mask[div:2*div,:,self.geom_dim:] = 0
            cond_mask[2*div:3*div, :, 2:] = 0
        else:
            print('Not a valid conditioning mask!')
            return
        return cond_mask
    
    def loss(self, pred, gt, seq_len):
        loss = 0
        for i in range(pred.shape[0]):
            L = seq_len[i]
            loss += self.loss_fcn(pred[i,:L], gt[i,:L])
        return loss / pred.shape[0]

    def additional_losses(self, cond_mask, ut, vt, loss):
        self.log("flow_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True) 
        if self.add_loss=='l1_loss':
            add_loss = torch.nn.functional.l1_loss(cond_mask*vt, cond_mask*ut)
        elif self.add_loss=='geom_l1_loss':
            add_loss = torch.nn.functional.l1_loss(cond_mask[...,:self.geom_dim]*ut[...,:self.geom_dim], cond_mask[...,:self.geom_dim]*vt[...,:self.geom_dim])
        else:
            print('add_loss not found!')
            add_loss = 0
        self.log(self.add_loss, add_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True) 
        loss += self.add_loss_weight * add_loss 
        return loss

    def inference(self, batch):
        return batch

    def generate_samples(self, batch):
        geom_pred = self.inference(batch)
        self.gen_data['bbox'].append(geom_pred.detach())
        self.gen_data['label'].append(batch['type'].detach())
        self.gen_data['seq_lens'].append(batch['length'].detach())
        pad_mask = torch.zeros(geom_pred.shape[:2], device=self.device, dtype=bool)
        for i, L in enumerate(batch['length']):
            pad_mask[i, :L] = True
        self.gen_data['pad_mask'].append(pad_mask)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 10000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = self.pe[:, :x.shape[1]]
        return self.dropout(x)

class TimePositionalEncoding(PositionalEncoding):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 10000):
        super().__init__(d_model, dropout, max_len)

    def forward(self, t: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = self.pe[:, t].movedim(0, 1)
        return self.dropout(x)