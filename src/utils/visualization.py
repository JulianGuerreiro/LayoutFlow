from typing import Any, Dict
import torch
import seaborn as sns

from PIL import Image, ImageDraw, ImageOps

def draw_layout(layout, features, num_colors=6, format='xywh', background_img=None, square=True):
    '''
    layout (S, 4): layout bbox given as (x, y, w, h) and in range (0, 1)
    features (S, 1): one-dimensional features that determine the color 
    '''
    colors = gen_colors(num_colors)

    if background_img:
        img = background_img
    else:
        if square:
            img = Image.new('RGB', (256, 256), color=(255, 255, 255))
        else:
            img = Image.new('RGB', (256, int(4/3*256)), color=(255, 255, 255))
    draw = ImageDraw.Draw(img, 'RGBA') 
    layout = torch.clip(layout, 0, 1)
    if format == 'ltwh':
        box = torch.stack([layout[:,0], layout[:,1], layout[:,2]+layout[:,0], layout[:,3]+layout[:,1]], dim=1)
    elif format == 'xywh':
        box = torch.stack([layout[:,0]-layout[:,2]/2, layout[:,1]-layout[:,3]/2, layout[:,0]+layout[:,2]/2, layout[:,1]+layout[:,3]/2], dim=1)
    elif format == 'ltrb':
        box = torch.stack([layout[:,0], layout[:,1], torch.maximum(layout[:,0], layout[:,2]), torch.maximum(layout[:,1], layout[:,3])], dim=1)
    else:
        print(f"Error: {format} format not supported.")
    box = 255*torch.clamp(box, 0, 1)

    for i in range(len(layout)):
        x1, y1, x2, y2 = box[i]
        if not square:
            y1 = int(4/3*y1)
            y2 = int(4/3*y2)
        cat = features[i]-1
        col = colors[cat] if 0 <= cat < len(colors) else [0, 0, 0]
        if cat < 0:
            continue
        draw.rectangle([x1, y1, x2, y2],
                        outline=tuple(col) + (200,),
                        fill=tuple(col) + (64,),
                        width=2)

    # Add border around image
    img = ImageOps.expand(img, border=2)
    return img

def gen_colors(num_colors):
    """
    Generate uniformly distributed `num_colors` colors
    :param num_colors:
    :return:
    """
    palette = sns.color_palette("husl", num_colors)
    rgb_triples = [[int(x[0]*255), int(x[1]*255), int(x[2]*255)] for x in palette]
    return rgb_triples