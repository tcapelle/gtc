import math

import matplotlib.pyplot as plt
from PIL import Image

import numpy as np

def _fig_bounds(x):
    r = x//32
    return min(5, max(1,r))

def show_image(im, ax=None, figsize=None, title=None, **kwargs):
    "Show a PIL or PyTorch image on `ax`."
    cmap=None
    # Handle pytorch axis order
    if hasattr(im, "device"):
        im = im.data.cpu()
        if im.shape[0]<5: im=im.permute(1,2,0)
    elif not isinstance(im, np.ndarray): 
        im=np.array(im)
    # Handle 1-channel images
    if im.shape[-1]==1: 
        cmap = "gray"
        im=im[...,0]
    
    if figsize is None: 
        figsize = (_fig_bounds(im.shape[0]), _fig_bounds(im.shape[1]))
    if ax is None: 
        _,ax = plt.subplots(figsize=figsize)
    ax.imshow(im, cmap=cmap, **kwargs)
    if title is not None: 
        ax.set_title(title)
    ax.axis('off')
    return ax

def show_images(ims, nrows=1, ncols=None, titles=None, **kwargs):
    "Show all images `ims` as subplots with `rows` using `titles`."
    if ncols is None: 
        ncols = int(math.ceil(len(ims)/nrows))
    if titles is None: 
        titles = [None]*len(ims)
    axs = plt.subplots(nrows, ncols, **kwargs)[1].flat
    for im,t,ax in zip(ims, titles, axs): 
        show_image(im, ax=ax, title=t)