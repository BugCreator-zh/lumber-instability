from random import random
import matplotlib.pyplot as plt
from sklearn import manifold, datasets
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
import seaborn as sns
import os
from PIL import Image
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def plot_tsne3(X,y,save_path,name1="",model_name=""):
    '''t-SNE'''
    # tsne = manifold.TSNE(n_components=2, random_state=0)
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    #tsne = PCA(n_components=2,random_state=0)
    labels_p = [(True if value == 1 else False) for value in y]
    labels_n = [(True if value == 0 else False) for value in y]
    labels_u = [(True if value == 2 else False) for value in y]

    X_tsne = tsne.fit_transform(X)
    colours = ListedColormap(['r', 'b', 'g', 'y', 'm'])
    #print("Org data dimension is {}. Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))
    color1 = ['red','blue','yellow']
    '''嵌入空间可视化'''
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
    plt.figure(figsize=(8, 8))

    X_p = X_norm[labels_p]
    X_n = X_norm[labels_n]
    X_u = X_norm[labels_u]
    # for i in range(X_p.shape[0]):
    p1 = plt.scatter(X_p[:, 0], X_p[:, 1], color=color1[0], alpha=0.6, s=8,label="pos")
    p2 = plt.scatter(X_n[:, 0], X_n[:, 1], color=color1[1], alpha=0.6, s=8,label="neg")
    p3 = plt.scatter(X_u[:, 0], X_u[:, 1], color=color1[2], alpha=0.6, s=8,label="unl")
        # ps1.append(p1)
        # ps2.append(p2)

    plt.title("t-SNE "+model_name+" "+name1)
    #plt.xticks([])
    #plt.yticks([])

    plt.legend()
    plt.savefig(save_path + name1 + "_" + model_name + ".jpg")
    #plt.show()

def show_conclusion(train_loss,train_score,test_score,save_path):
    iteration = range(len(train_loss))
    plt.figure()
    plt.plot(iteration,train_loss)
    plt.plot(iteration,train_score)
    plt.plot(iteration,test_score)
    plt.legend(['train_loss','train_score','test_score'])
    plt.savefig(save_path)