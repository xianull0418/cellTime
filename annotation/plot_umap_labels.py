import argparse
import os
import numpy as np
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_path', type=str, required=True)
    parser.add_argument('--true_label_path', type=str, required=True)
    parser.add_argument('--pred_label_path', type=str, required=True)
    parser.add_argument('--label_names_path', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='.')
    parser.add_argument('--prefix', type=str, default='umap')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    emb = np.load(args.embedding_path)
    y_true = np.load(args.true_label_path)
    y_pred = np.load(args.pred_label_path)
    label_names = None
    if args.label_names_path and os.path.exists(args.label_names_path):
        label_names = np.load(args.label_names_path)

    ad = sc.AnnData(emb)
    sc.pp.pca(ad, n_comps=min(50, emb.shape[1]))
    sc.pp.neighbors(ad, use_rep='X_pca')
    sc.tl.umap(ad)
    ad.obs['true_label'] = (label_names[y_true] if label_names is not None else y_true.astype(str))
    ad.obs['pred_label'] = (label_names[y_pred] if label_names is not None else y_pred.astype(str))

    fig_dir = os.path.join(args.save_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    # unified palette
    cats_true = list(ad.obs['true_label'].astype('category').cat.categories)
    cats_pred = list(ad.obs['pred_label'].astype('category').cat.categories)
    all_cats = sorted(list(set(cats_true) | set(cats_pred)))
    cmap = plt.get_cmap('tab20')
    color_map = {c: cmap(i % cmap.N) for i, c in enumerate(all_cats)}
    coords = ad.obsm['X_umap']
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax1, ax2 = axes
    for c in all_cats:
        m1 = (ad.obs['true_label'].astype(str).values == c)
        m2 = (ad.obs['pred_label'].astype(str).values == c)
        ax1.scatter(coords[m1, 0], coords[m1, 1], s=5, c=color_map[c], label=c)
        ax2.scatter(coords[m2, 0], coords[m2, 1], s=5, c=color_map[c], label=c)
    ax1.set_title('true')
    ax2.set_title('pred')
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=c, markerfacecolor=color_map[c], markersize=6) for c in all_cats]
    fig.legend(handles=handles, labels=all_cats, loc='lower center', bbox_to_anchor=(0.5, -0.02), ncol=min(5, len(all_cats)))
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    out_pdf = os.path.join(fig_dir, f'{args.prefix}_true_pred.pdf')
    out_png = os.path.join(fig_dir, f'{args.prefix}_true_pred.png')
    fig.savefig(out_pdf, dpi=150, bbox_inches='tight')
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    print(f'saved figure: {out_pdf}')
    print(f'saved figure: {out_png}')
    plt.close(fig)
    ad.write(os.path.join(args.save_dir, f'{args.prefix}_adata.h5ad'))

if __name__ == '__main__':
    main()