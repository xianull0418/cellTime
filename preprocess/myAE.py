#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import scanpy as sc
from matplotlib.pyplot import rc_context
import os
from pathlib import Path
import matplotlib.pyplot as plt
import gc
import sys
sys.path.append('../') # Path to AE folder
from AE import AutoEncoder, Trainer
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Training parameters')

    # 位置参数
    parser.add_argument('input_h5ad', type=str, help='Input h5ad file path')
    parser.add_argument('output_h5ad', type=str, help='Output h5ad file path')

    # 可选参数
    parser.add_argument('--seed', type=int, default=4232, help='Random seed (default: 4232)')
    parser.add_argument('--n_layers', type=int, default=1, help='Number of layers (default: 1)')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size (default: 128)')
    parser.add_argument('--dataset', type=str, default='TEDD', help="Dataset name (default: 'TEDD')")
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate (default: 1e-3)')
    parser.add_argument('--n_hidden', type=int, default=300, help='Number of hidden units (default: 300)')
    parser.add_argument('--n_latent', type=int, default=10, help='Number of latent dimensions (default: 10)')
    parser.add_argument('--mode', type=str, default='training', help="Mode of operation: training or loading (default: 'training')")
    parser.add_argument('--max_epoch', type=int, default=500, help='Maximum number of epochs (default: 500)')

    args = parser.parse_args()
    return args

def folder_dir(dataset:str='EMT',
         seed:int=42,
         n_latent:int=6,
         n_hidden:int=300,
         n_layers: int=1,
         activation: str = 'relu',
         dropout:float=0.2,
         weight_decay:float=1e-4,
         lr:float=1e-3,
         batch_size: int=32,):
    folder=Path('results/'+dataset+'_'+str(seed)+\
           '_'+str(n_latent)+'_'+str(n_layers)+'_'+str(n_hidden)+\
           '_'+str(dropout)+'_'+str(weight_decay)+'_'+str(lr)+'_'+str(batch_size)+'/')
    return folder

def generate_plots(folder,model, adata,seed,n_neighbors=10,min_dist=0.5,plots='umap'):
    model.eval()
    with torch.no_grad():
        X_latent_AE=model.get_latent_representation(torch.tensor(adata.X).type(torch.float32).to('cpu'))
    adata.obsm['X_AE']=X_latent_AE.detach().cpu().numpy()
    sc.pp.neighbors(adata, n_neighbors=n_neighbors,use_rep='X_AE')

    color=['time']
    if plots=='umap':
        sc.tl.umap(adata,random_state=seed,min_dist=min_dist)
        with rc_context({'figure.figsize': (8, 8*len(color))}):
            sc.pl.umap(adata, color=color,
                       legend_loc='on data',
                       legend_fontsize=12,
                       legend_fontoutline=2, )
        plt.savefig(str(folder) + '/umap.pdf')
        plt.close()
    elif plots=='embedding':
        with rc_context({'figure.figsize': (8*len(color), 8)}):
            sc.pl.embedding(adata, 'X_AE',color=color,
                       # legend_loc='on data',
                       legend_fontsize=12,
                       legend_fontoutline=2, )
            plt.legend(frameon=False)
            plt.xticks([plt.xlim()[0], 0., plt.xlim()[1]])
            plt.yticks([plt.ylim()[0], 0., plt.ylim()[1]])
        plt.savefig(str(folder) + '/embedding.pdf')
        plt.close()

def loss_plots(folder,model):
    fig,axs=plt.subplots(1, 1, figsize=(4, 4))
    axs.set_title('AE loss')
    axs.plot(model.history['epoch'], model.history['train_loss'])
    axs.plot(model.history['epoch'], model.history['val_loss'])
    plt.yscale('log')
    axs.legend(['train loss','val loss'])
    plt.savefig(str(folder)+'/loss.pdf')
    plt.close()

def main(dataset:str='EMT',
         seed:int=42,
         n_latent:int=6,
         n_hidden:int=300,
         n_layers: int=1,
         activation: str='relu',
         dropout:float=0.2,
         weight_decay:float=1e-4,
         lr:float=1e-3,
         max_epoch:int=500,
         batch_size: int=32,
         mode='training',
         input_h5ad:str=None
         ):

    adata = sc.read_h5ad(input_h5ad)
    X = adata.X

    model=AutoEncoder(in_dim=X.shape[1],
                      n_latent=n_latent,
                      n_hidden=n_hidden,
                      n_layers=n_layers,
                      activate_type=activation,
                      dropout=dropout,
                      norm=True,
                      seed=seed,)

    trainer=Trainer(model,X=X,
                    test_size=0.1,
                    lr=lr,
                    batch_size=batch_size,
                    weight_decay=weight_decay,
                    seed=seed)

    folder=folder_dir(dataset=dataset,
         seed=seed,
         n_latent=n_latent,
         n_hidden=n_hidden,
         n_layers=n_layers,
         dropout=dropout,
         activation=activation,
         weight_decay=weight_decay,
         lr=lr,
         batch_size=batch_size,)

    if mode=='training':
        print('training the model')
        trainer.train(max_epoch=max_epoch,patient=30)

        # model.eval()
        if not os.path.exists(folder):
            folder.mkdir(parents=True)
        torch.save({
            'func_state_dict': model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'loss_history':trainer.model.history,
        }, os.path.join(folder,'model.pt'))
    elif mode=='loading':
        print('loading the model')
        check_pt = torch.load(os.path.join(folder, 'model.pt'))

        model.load_state_dict(check_pt['func_state_dict'])
        trainer.optimizer.load_state_dict(check_pt['optimizer_state_dict'])
        model.history=check_pt['loss_history']
    return model,trainer,adata,folder


if __name__ == '__main__':
    args = parse_args()

    model, trainer, adata, folder=main(
            dataset=args.dataset,
            seed=args.seed,
            n_layers=args.n_layers,
            n_latent=args.n_latent,
            n_hidden=args.n_hidden,
            activation='relu',
            lr=args.lr,
            batch_size=args.batch_size,
            max_epoch=args.max_epoch,
            mode=args.mode,
            input_h5ad=args.input_h5ad
            )

    model=model.to('cpu')

    generate_plots(folder,model, adata, args.seed,n_neighbors=20,min_dist=0.5,plots='embedding')
    generate_plots(folder,model, adata, args.seed,n_neighbors=20,min_dist=0.5,plots='umap')
    loss_plots(folder,model)

    adata.write(args.output_h5ad, compression="gzip")

