import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
from glob import glob


def show_fg():
    normal_fg = np.loadtxt("normal_fg.txt")
    hard_fg = np.loadtxt("hard_fg.txt")
    normal_fg = manifold.TSNE(n_components=2, init='pca', random_state=42).fit_transform(normal_fg)
    hard_fg = manifold.TSNE(n_components=2, init='pca', random_state=42).fit_transform(hard_fg)
    nf_min, nf_max = normal_fg.min(0), normal_fg.max(0)
    normal_fg = (normal_fg - nf_min) / (nf_max - nf_min)
    hf_min, hf_max = hard_fg.min(0), hard_fg.max(0)
    hard_fg = (hard_fg - hf_min) / (hf_max - hf_min)
    plt.figure(figsize=(4, 4))
    plt.scatter(normal_fg[:, 0], normal_fg[:, 1], 300, color='blue', label='Normal Fg', marker='*')
    plt.scatter(hard_fg[:, 0], hard_fg[:, 1], 300, color='green', label='Hard Fg', marker='*')
    plt.legend(loc='upper left')
    plt.show()


def show_bg():
    normal_bg = np.loadtxt("normal_bg.txt")
    hard_bg = np.loadtxt("hard_bg.txt")
    normal_bg = manifold.TSNE(n_components=2, init='pca', random_state=42).fit_transform(normal_bg)
    hard_bg = manifold.TSNE(n_components=2, init='pca', random_state=42).fit_transform(hard_bg)
    nb_min, nb_max = normal_bg.min(0), normal_bg.max(0)
    normal_bg = (normal_bg - nb_min) / (nb_max - nb_min)
    hb_min, hb_max = hard_bg.min(0), hard_bg.max(0)
    hard_bg = (hard_bg - hb_min) / (hb_max - hb_min)
    plt.figure(figsize=(4, 4))
    plt.scatter(normal_bg[:, 0], normal_bg[:, 1], 300, color='black', label='Normal Bg', marker='*')
    plt.scatter(hard_bg[:, 0], hard_bg[:, 1], 300, color='red', label='Hard Bg', marker='*')
    plt.legend(loc='upper left')
    plt.show()


show_fg()
# show_bg()







