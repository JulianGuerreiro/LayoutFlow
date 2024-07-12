from typing import Dict, List, Optional, Tuple, Union
from torch import FloatTensor

import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from itertools import chain

Feats = Union[FloatTensor, List[FloatTensor]]
Layout = Tuple[np.ndarray, np.ndarray]

# from utils import convert_xywh_to_ltrb
def convert_xywh_to_ltrb(bbox: Union[np.ndarray, FloatTensor]):
    xc, yc, w, h = bbox
    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2
    return [x1, y1, x2, y2]

def compute_iou(
    box_1: Union[np.ndarray, FloatTensor],
    box_2: Union[np.ndarray, FloatTensor],
    generalized: bool = False,
) -> Union[np.ndarray, FloatTensor]:
    # box_1: [N, 4]  box_2: [N, 4]

    if isinstance(box_1, np.ndarray):
        lib = np
    else:
        lib = torch

    l1, t1, r1, b1 = convert_xywh_to_ltrb(box_1.T)
    l2, t2, r2, b2 = convert_xywh_to_ltrb(box_2.T)
    a1, a2 = (r1 - l1) * (b1 - t1), (r2 - l2) * (b2 - t2)

    # intersection
    l_max = lib.maximum(l1, l2)
    r_min = lib.minimum(r1, r2)
    t_max = lib.maximum(t1, t2)
    b_min = lib.minimum(b1, b2)
    cond = (l_max < r_min) & (t_max < b_min)
    ai = lib.where(cond, (r_min - l_max) * (b_min - t_max), lib.zeros_like(a1[0]))

    au = a1 + a2 - ai
    iou = ai / au

    if not generalized:
        return iou

    # outer region
    l_min = lib.minimum(l1, l2)
    r_max = lib.maximum(r1, r2)
    t_min = lib.minimum(t1, t2)
    b_max = lib.maximum(b1, b2)
    ac = (r_max - l_min) * (b_max - t_min)

    giou = iou - (ac - au) / ac

    return giou

def __compute_maximum_iou_for_layout(layout_1: Layout, layout_2: Layout) -> float:
    score = 0.0
    (bi, li), (bj, lj) = layout_1, layout_2
    N = len(bi)
    for l in list(set(li.tolist())):
        _bi = bi[np.where(li == l)]
        _bj = bj[np.where(lj == l)]
        n = len(_bi)
        ii, jj = np.meshgrid(range(n), range(n))
        ii, jj = ii.flatten(), jj.flatten()
        iou = compute_iou(_bi[ii], _bj[jj]).reshape(n, n)
        if True in torch.isnan(iou):
            continue
        # note: maximize is supported only when scipy >= 1.4
        ii, jj = linear_sum_assignment(iou, maximize=True)
        score += iou[ii, jj].sum().item()
    return score / N


def __compute_maximum_iou(layouts_1_and_2: Tuple[List[Layout]]) -> np.ndarray:
    layouts_1, layouts_2 = layouts_1_and_2
    N, M = len(layouts_1), len(layouts_2)
    ii, jj = np.meshgrid(range(N), range(M))
    ii, jj = ii.flatten(), jj.flatten()
    scores = np.asarray(
        [
            __compute_maximum_iou_for_layout(layouts_1[i], layouts_2[j])
            for i, j in zip(ii, jj)
        ]
    ).reshape(N, M)
    ii, jj = linear_sum_assignment(scores, maximize=True)
    return scores[ii, jj]


def __get_cond2layouts(layout_list: List[Layout]) -> Dict[str, List[Layout]]:
    out = dict()
    for bs, ls in layout_list:
        cond_key = str(sorted(ls.tolist()))
        if cond_key not in out.keys():
            out[cond_key] = [(bs, ls)]
        else:
            out[cond_key].append((bs, ls))
    return out

def compute_maximum_iou(
    layouts_1,
    layouts_2,
    format = 'xywh',
):
    """
    Computes Maximum IoU [Kikuchi+, ACMMM'21]
    """
    c2bl_1 = __get_cond2layouts(layouts_1)
    keys_1 = set(c2bl_1.keys())
    c2bl_2 = __get_cond2layouts(layouts_2)
    keys_2 = set(c2bl_2.keys())
    keys = list(keys_1.intersection(keys_2))
    args = [(c2bl_1[key], c2bl_2[key]) for key in keys]
    # to check actual number of layouts for evaluation
    # ans = 0
    # for x in args:
    #     ans += len(x[0])
    scores = [__compute_maximum_iou(a) for a in args]
    scores = np.asarray(list(chain.from_iterable(scores)))
    if len(scores) == 0:
        return 0.0
    else:
        return scores.mean().item()


def compute_overlap(bbox, mask, format='xywh'):
    # Attribute-conditioned Layout GAN
    # 3.6.3 Overlapping Loss

    bbox = bbox.masked_fill(~mask.unsqueeze(-1), 0)
    bbox = bbox.permute(2, 0, 1)


    if format == 'xywh':
        l1, t1, r1, b1 = convert_xywh_to_ltrb(bbox.unsqueeze(-1))
        l2, t2, r2, b2 = convert_xywh_to_ltrb(bbox.unsqueeze(-2))
    elif format == 'ltrb':
        l1, t1, r1, b1 = bbox.unsqueeze(-1)
        l2, t2, r2, b2 = bbox.unsqueeze(-2)
    else:
        print(f'{format} format not supported.')
        return

    a1 = (r1 - l1) * (b1 - t1)

    # intersection
    l_max = torch.maximum(l1, l2)
    r_min = torch.minimum(r1, r2)
    t_max = torch.maximum(t1, t2)
    b_min = torch.minimum(b1, b2)
    cond = (l_max < r_min) & (t_max < b_min)
    ai = torch.where(cond, (r_min - l_max) * (b_min - t_max),
                     torch.zeros_like(a1[0]))

    diag_mask = torch.eye(a1.size(1), dtype=torch.bool,
                          device=a1.device)
    ai = ai.masked_fill(diag_mask, 0)

    ar = ai / a1
    ar = torch.from_numpy(np.nan_to_num(ar.numpy()))
    score = torch.from_numpy(
        np.nan_to_num((ar.sum(dim=(1, 2)) / mask.float().sum(-1)).numpy())
    )
    return (score).mean().item()


def compute_alignment(bbox, mask, format='xywh', output_torch=False):
    # Attribute-conditioned Layout GAN
    # 3.6.4 Alignment Loss

    bbox = bbox.permute(2, 0, 1)
    if format == 'xywh':
        xl, yt, xr, yb = convert_xywh_to_ltrb(bbox)
    elif format == 'ltrb':
        xl, yt, xr, yb = bbox
    else:
        print(f'{format} format not supported.')
        return
    xc = (xr + xl) / 2
    yc = (yt + yb) / 2
    X = torch.stack([xl, xc, xr, yt, yc, yb], dim=1)

    X = X.unsqueeze(-1) - X.unsqueeze(-2)
    idx = torch.arange(X.size(2), device=X.device)
    X[:, :, idx, idx] = 1.
    X = X.abs().permute(0, 2, 1, 3)
    X[~mask] = 1.
    X = X.min(-1).values.min(-1).values
    X.masked_fill_(X.eq(1.), 0.)

    X = -torch.log(1 - X)
    if not output_torch:
        score = torch.from_numpy(np.nan_to_num((X.sum(-1) / mask.float().sum(-1)))).numpy()
    else:
        score = torch.nan_to_num(X.sum(-1) / mask.float().sum(-1))
    return (score).mean().item()


def compute_overlap_ignore_bg(bbox, label, mask, format='xywh'):
    mask = torch.where(label == 4,  False, mask)  # List Item
    mask = torch.where(label == 9,  False, mask)  # Card
    mask = torch.where(label == 11, False, mask)  # Background Image
    mask = torch.where(label == 17, False, mask)  # Modal

    return compute_overlap(bbox, mask, format)