import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import scanpy as sc
from scipy.sparse import issparse
import pickle
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix, roc_auc_score

def stratified_split(y, test_ratio=0.2, val_ratio=0.1, seed=0):
    rng = np.random.RandomState(seed)
    classes = np.unique(y)
    train_idx = []
    val_idx = []
    test_idx = []
    for c in classes:
        idx_c = np.where(y == c)[0]
        idx_c = rng.permutation(idx_c)
        n_c = len(idx_c)
        n_test = int(round(n_c * test_ratio))
        n_val = int(round((n_c - n_test) * val_ratio))
        test_idx.append(idx_c[:n_test])
        val_idx.append(idx_c[n_test:n_test + n_val])
        train_idx.append(idx_c[n_test + n_val:])
    train_idx = np.concatenate(train_idx)
    val_idx = np.concatenate(val_idx)
    test_idx = np.concatenate(test_idx)
    return train_idx, val_idx, test_idx

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def load_features(path, hvg_num=2000, pre_normalized=False, hvg_flavor='seurat'):
    if path.endswith('.h5ad'):
        adata = sc.read_h5ad(path)
        if issparse(adata.X):
            X = adata.X.toarray()
        else:
            X = adata.X
        ad = sc.AnnData(X)
        if not pre_normalized:
            sc.pp.normalize_total(ad)
            sc.pp.log1p(ad)
        sc.pp.highly_variable_genes(ad, n_top_genes=hvg_num, flavor=hvg_flavor)
        ad = ad[:, ad.var['highly_variable']]
        return ad.X, None
    elif path.endswith('.npy'):
        X = np.load(path)
        return X, None
    elif path.endswith('.csv'):
        import pandas as pd
        X = pd.read_csv(path, index_col=0).values
        return X, None
    else:
        raise ValueError('Unsupported input format')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--label_path', type=str, required=True)
    parser.add_argument('--label_names_path', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='.')
    parser.add_argument('--hvg_num', type=int, default=2000)
    parser.add_argument('--pre_normalized', action='store_true')
    parser.add_argument('--hvg_flavor', type=str, default='seurat')
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--pkl_name', type=str, default='zheng-emb-2mlp.pkl')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    X, _ = load_features(args.data_path, hvg_num=args.hvg_num, pre_normalized=args.pre_normalized, hvg_flavor=args.hvg_flavor)
    y = np.load(args.label_path)
    if X.shape[0] != y.shape[0]:
        raise ValueError('Feature and label length mismatch')
    train_idx, val_idx, test_idx = stratified_split(y, test_ratio=args.test_ratio, val_ratio=args.val_ratio, seed=args.seed)

    num_classes = int(np.max(y)) + 1
    in_dim = X.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_val = X[val_idx]
    y_val = y[val_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]

    X_t = torch.tensor(X, dtype=torch.float32)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds = TensorDataset(X_val_t, y_val_t)
    test_ds = TensorDataset(X_test_t, y_test_t)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    model = MLP(in_dim, args.hidden_dim, num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                pred = logits.argmax(dim=1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)
        val_acc = correct / max(total, 1)
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        logits_all = []
        for i in range(0, X_t.size(0), args.batch_size):
            xb = X_t[i:i + args.batch_size].to(device)
            logits = model(xb)
            logits_all.append(logits.cpu())
        logits_all = torch.cat(logits_all, dim=0)
        probs_all = torch.softmax(logits_all, dim=1).numpy()
        preds_all = logits_all.argmax(dim=1).numpy()

    emb_path = os.path.join(args.save_dir, 'cells_embedding.npy')
    np.save(emb_path, X)
    pred_path = os.path.join(args.save_dir, 'pred_labels.npy')
    np.save(pred_path, preds_all)

    pkl_path = os.path.join(args.save_dir, args.pkl_name)
    with open(pkl_path, 'wb') as f:
        pickle.dump({'emb': probs_all, 'label': preds_all, 'true': y}, f)

    test_correct = 0
    test_total = 0
    with torch.no_grad():
        xb = X_test_t.to(device)
        yb = y_test_t.to(device)
        logits = model(xb)
        pred = logits.argmax(dim=1)
        test_correct = (pred == yb).sum().item()
        test_total = yb.size(0)
    test_acc = test_correct / max(test_total, 1)
    with torch.no_grad():
        test_probs = torch.softmax(logits, dim=1).cpu().numpy()
    y_test_np = y_test_t.cpu().numpy()
    y_pred_test_np = pred.cpu().numpy()

    

    metrics_path = os.path.join(args.save_dir, 'metrics.txt')
    with open(metrics_path, 'w') as wf:
        wf.write(f'test_acc\t{test_acc}\n')
        try:
            if args.label_names_path is not None and os.path.exists(args.label_names_path):
                label_names = np.load(args.label_names_path)
                rep = classification_report(y_test_np, y_pred_test_np, target_names=label_names, digits=4)
            else:
                rep = classification_report(y_test_np, y_pred_test_np, digits=4)
            wf.write(rep + "\n")
        except Exception:
            pass
        try:
            wf.write(f'accuracy\t{accuracy_score(y_test_np, y_pred_test_np)}\n')
            wf.write(f'macro_f1\t{f1_score(y_test_np, y_pred_test_np, average="macro")}\n')
            wf.write(f'weighted_f1\t{f1_score(y_test_np, y_pred_test_np, average="weighted")}\n')
        except Exception:
            pass
        try:
            cm = confusion_matrix(y_test_np, y_pred_test_np)
            wf.write('confusion_matrix\n')
            wf.write(np.array2string(cm) + "\n")
        except Exception:
            pass
        try:
            roc_macro = roc_auc_score(y_test_np, test_probs, multi_class='ovr', average='macro')
            wf.write(f'macro_roc_auc\t{roc_macro}\n')
        except Exception:
            pass

if __name__ == '__main__':
    main()