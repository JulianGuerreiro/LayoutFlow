from typing import Dict, List, Optional, Tuple, Union
from torch import BoolTensor, FloatTensor

import torch
import numpy as np
import matplotlib.pyplot as plt

def convert_xywh_to_ltrb(bbox: Union[np.ndarray, FloatTensor]):
    xc, yc, w, h = bbox
    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2
    return [x1, y1, x2, y2]

def convert_bbox(bbox: FloatTensor, conv='xywh->ltwh'):
    '''
    bbox (*, 4): bounding box
    conv (str): conversion (e.g. 'xywh->lrwh')
    '''
    if conv.split('->')[0] == conv.split('->')[1]:
        return bbox

    a, b, c, d = bbox.movedim(-1, 0)
    if conv == 'xywh->ltwh':
        l = a - c/2
        t = b - d/2
        w, h = c, d
        return torch.stack([l, t, w, h], dim=-1)
    elif conv == 'ltwh->xywh':
        x = a + c/2
        y = b + d/2
        w, h = c, d
        return torch.stack([x, y, w, h], dim=-1)
    elif conv == 'xywh->ltrb':
        l = a - c/2
        t = b - d/2
        r = a + c/2
        b = b + d/2
        return torch.stack([l, t, r, b], dim=-1)
    elif conv == 'ltrb->xywh':
        x = (a + c)/2
        y = (b + d)/2
        w = (c - a)
        h = (d - b)
        return torch.stack([x, y, w, h], dim=-1)
    elif conv == 'ltwh->ltrb':
        l, t = a, b
        r = a + c
        b = b + d
        return torch.stack([l, t, r, b], dim=-1)
    elif conv == 'ltrb->ltwh':
        l, t = a, b
        w = a - c
        h = b - d
        return torch.stack([l, t, w, h], dim=-1)
    else:
        print("conversion not found!")

def plot_trajectories(traj, categories=False, num_cat=0, connect_points=True):
    """Plot trajectories of some selected samples."""
    n = 2000
    fig = plt.figure(figsize=(6, 6), dpi=200)
    if not categories:
        if connect_points:
            plt.plot(traj[:, :n, 0], traj[:, :n, 1],'--', linewidth=0.8, c='olive')
            plt.plot(traj[0, :n, 0], traj[0, :n, 1], 'v', alpha=1, c='black')
            plt.plot(traj[-1, :n, 0], traj[-1, :n, 1], 'X', alpha=1, c='blue')
            # Canvas Borders
            plt.plot(np.ones(2), np.array([0, 1]), alpha=0.8, c="black")
            plt.plot(np.zeros(2),np.array([0, 1]), alpha=0.8, c="black")
            plt.plot(np.array([0, 1]), np.zeros(2), alpha=0.8, c="black")
            plt.plot(np.array([0, 1]), np.ones(2), alpha=0.8, c="black")
        else:
            plt.scatter(traj[0, :n, 0], traj[0, :n, 1], s=10, alpha=0.8, c="black")
            plt.scatter(traj[:, :n, 0], traj[:, :n, 1], s=0.2, alpha=0.2, c="olive")
            plt.scatter(traj[-1, :n, 0], traj[-1, :n, 1], s=4, alpha=1, c="blue")
            # Canvas Borders
            plt.scatter(np.ones(50), np.arange(0,1,0.02), s=4, alpha=1, c="gray")
            plt.scatter(np.zeros(50), np.arange(0,1,0.02), s=4, alpha=1, c="gray")
            plt.scatter(np.arange(0,1,0.02), np.zeros(50), s=4, alpha=1, c="gray")
            plt.scatter(np.arange(0,1,0.02), np.ones(50), s=4, alpha=1, c="gray")
        plt.axis('scaled')
        plt.legend(["Prior sample", "Flow", "Prediction"])
    else:
        plt.hlines(np.arange(1, num_cat), xmin=0, xmax=traj.shape[0], color='gray', linestyle='--')
        plt.xlabel('t (Time)')
        plt.ylabel('c (Categories)')
        plt.plot(traj)
    return fig
