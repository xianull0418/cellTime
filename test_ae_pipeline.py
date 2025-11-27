import os
import shutil
import pandas as pd
import scanpy as sc
import numpy as np
from pathlib import Path
import torch
import sys
from omegaconf import OmegaConf

# Add project root to path
sys.path.append(os.getcwd())

from dataset.ae_data_process import process_data
from models.ae import AESystem
from dataset import StaticCellDataset

def test_pipeline():
    print("Starting Pipeline Verification (scBank)...")
    
    # Setup paths
    test_dir = Path("test_data_process_scbank")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir()
    
    h5ad_dir = test_dir / "h5ad"
    h5ad_dir.mkdir()
    
    processed_dir = test_dir / "processed"
    
    # 1. Create Dummy Data
    print("\n[1] Creating Dummy Data...")
    n_genes = 100
    n_cells_train = 500
    n_cells_ood = 100
    
    # Train Data
    X_train = np.random.randint(0, 100, (n_cells_train, n_genes))
    obs_train = pd.DataFrame(index=[f"cell_{i}" for i in range(n_cells_train)])
    var_train = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])
    adata_train = sc.AnnData(X=X_train, obs=obs_train, var=var_train)
    train_h5ad = h5ad_dir / "train_data.h5ad"
    adata_train.write(train_h5ad)
    
    # OOD Data
    X_ood = np.random.randint(0, 100, (n_cells_ood, n_genes))
    obs_ood = pd.DataFrame(index=[f"ood_cell_{i}" for i in range(n_cells_ood)])
    adata_ood = sc.AnnData(X=X_ood, obs=obs_ood, var=var_train)
    ood_h5ad = h5ad_dir / "ood_data.h5ad"
    adata_ood.write(ood_h5ad)
    
    # Create gene_order.tsv (optional for test, but good to have)
    gene_order_path = Path("gene_order.tsv")
    # Backup existing if any
    if gene_order_path.exists():
        shutil.copy(gene_order_path, gene_order_path.with_suffix(".bak"))
    
    # Write dummy gene order
    with open(gene_order_path, "w") as f:
        for g in var_train.index:
            f.write(f"{g}\n")
    
    # Create CSV
    csv_data = {
        "file_path": [str(train_h5ad), str(ood_h5ad)],
        "full_validation_dataset": [0, 1],
        "cell_count": [n_cells_train, n_cells_ood]
    }
    csv_path = test_dir / "data_info.csv"
    pd.DataFrame(csv_data).to_csv(csv_path, index=False)
    print(f"Created dummy data at {test_dir}")
    
    # 2. Run Processing Script
    print("\n[2] Running dataset/ae_data_process.py...")
    try:
        process_data(
            csv_path=str(csv_path),
            output_dir=str(processed_dir),
            min_genes=0, 
            num_workers=1 # Use 1 worker to avoid multiprocessing issues in simple test
        )
    except Exception as e:
        print(f"FAILED: Processing script error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Verify outputs
    train_scbank = processed_dir / "train.scbank"
    ood_scbank = processed_dir / "ood.scbank"
    
    if not (train_scbank / "manifest.json").exists():
        print(f"FAILED: Train scBank manifest not found at {train_scbank}")
        return
    if not (ood_scbank / "manifest.json").exists():
        print(f"FAILED: OOD scBank manifest not found at {ood_scbank}")
        return
    print("Confirmed: scBank datasets created")

    # 3. Test Model Initialization
    print("\n[3] Testing AESystem Initialization...")
    try:
        # Create config using OmegaConf
        cfg = OmegaConf.create({
            "data": {
                "dataset_type": "scbank",
                "scbank_path": {
                    "train": str(train_scbank),
                    "ood": str(ood_scbank),
                },
                "process": {
                    "train_ratio": 0.8
                },
                "num_workers": 0,
                "pin_memory": False
            },
            "model": {
                "n_genes": n_genes,
                "latent_dim": 10,
                "hidden_dim": 20,
                "dropout_rate": 0.1
            },
            "training": {
                "batch_size": 16,
                "reconstruction_loss": "mse",
                "learning_rate": 1e-3,
                "weight_decay": 0,
                "scheduler": None
            },
            "logging": {
                "output_dir": str(test_dir / "logs")
            },
            "accelerator": {
                "accelerator": "cpu",
                "devices": 1
            }
        })
        
        model = AESystem(cfg)
        model.setup(stage="fit")
        
        if model.train_dataset is None:
            print("FAILED: train_dataset is None")
            return
        if model.val_dataset is None:
            print("FAILED: val_dataset is None")
            return
        if model.ood_dataset is None:
            print("FAILED: ood_dataset is None")
            return
            
        print(f"Train size: {len(model.train_dataset)}")
        print(f"Val size: {len(model.val_dataset)}")
        print(f"OOD size: {len(model.ood_dataset)}")
        
        # Check split ratio
        total_train = len(model.train_dataset) + len(model.val_dataset)
        ratio = len(model.train_dataset) / total_train
        print(f"Actual train ratio: {ratio:.2f}")
        
        print("Confirmed: AESystem initialized and datasets loaded")
        
    except Exception as e:
        print(f"FAILED: Model initialization error: {e}")
        import traceback
        traceback.print_exc()
        return
        
    print("\nSUCCESS: All verification steps passed!")
    
    # Cleanup
    shutil.rmtree(test_dir)
    # Restore gene_order if backed up
    if Path("gene_order.tsv.bak").exists():
        shutil.move("gene_order.tsv.bak", "gene_order.tsv")
    elif Path("gene_order.tsv").exists():
        os.remove("gene_order.tsv")

if __name__ == "__main__":
    test_pipeline()
