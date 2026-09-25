import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from tqdm import tqdm  # <-- ADDED TQDM

# Import the original model and losses
from rin_encodermap.encodermap_model import EncoderMapNet, auto_cost, sketch_cost, l2_regularization

# --- CONFIGURATION ---
NPY_FILE = "closeness_fingerprints.npy"
INDICES_DIR = "fold_indices"
OUTPUT_DIR = "cross_val_results"
WINDOW_SIZE = 128
INPUT_DIM = WINDOW_SIZE * WINDOW_SIZE # 16384
BOTTLENECK_DIM = 32
BATCH_SIZE = 512
EPOCHS = 10
LEARNING_RATE = 1e-4
CONTACT_THRESHOLD = 0.1 # Values < 0.1 are considered "Interacting" (Contact)

class MappedDataset(torch.utils.data.Dataset):
    """Memory-safe dataset that reads from disk using mmap."""
    def __init__(self, data_path, mask_path):
        self.data = np.load(data_path, mmap_mode='r')
        self.mask = np.load(mask_path)
        self.indices = np.where(self.mask)[0]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Convert numpy float64 to torch float32 on the fly
        return torch.tensor(self.data[self.indices[idx]], dtype=torch.float32)

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    # ADDED TQDM
    for batch in tqdm(dataloader, desc="  Train Batches", leave=False):
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Original EncoderMap methodology: code, recon = model(x)
        code, recon = model(batch)
        
        # Calculate losses (using original auto_cost + sketch_cost)
        loss_auto = auto_cost(batch, recon)
        loss_sketch = sketch_cost(batch, code, (0.5, 6, 6), (1.0, 2, 6))
        loss_reg = l2_regularization(model) * 1e-5
        
        loss = loss_auto + loss_sketch + loss_reg
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch.size(0)
    return total_loss / len(dataloader.dataset)

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        # ADDED TQDM
        for batch in tqdm(dataloader, desc="  Eval Batches", leave=False):
            batch = batch.to(device)
            code, recon = model(batch)
            
            loss_auto = auto_cost(batch, recon)
            loss_sketch = sketch_cost(batch, code, (0.5, 6, 6), (1.0, 2, 6))
            loss = loss_auto + loss_sketch
            
            total_loss += loss.item() * batch.size(0)
            
            # Store for accuracy/confusion matrix
            all_preds.append(recon.cpu().numpy())
            all_targets.append(batch.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    
    # Calculate "Accuracy" and Confusion Matrix
    # We binarize: < CONTACT_THRESHOLD is "Contact" (1), else "Non-contact" (0)
    preds_flat = np.concatenate(all_preds).flatten()
    targets_flat = np.concatenate(all_targets).flatten()
    
    pred_binary = (preds_flat < CONTACT_THRESHOLD).astype(int)
    target_binary = (targets_flat < CONTACT_THRESHOLD).astype(int)
    
    acc = accuracy_score(target_binary, pred_binary)
    cm = confusion_matrix(target_binary, pred_binary)
    
    return avg_loss, acc, cm, preds_flat

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Using device: {device}")
    
    # Verify data exists
    if not os.path.exists(NPY_FILE):
        print(f"❌ {NPY_FILE} not found. Run run_pipeline.py first.")
        return

    fold_metrics = {'train_loss': [], 'val_loss': [], 'test_loss': [], 'val_acc': [], 'test_acc': []}
    
    # ADDED TQDM FOR FOLDS
    for k in tqdm(range(1, 11), desc="Cross-Validation Folds"):
        print(f"\n{'='*20} FOLD {k}/10 {'='*20}")
        
        # 1. Create DataLoaders (Memory Mapped)
        train_ds = MappedDataset(NPY_FILE, os.path.join(INDICES_DIR, f"train_mask_fold{k}.npy"))
        val_ds = MappedDataset(NPY_FILE, os.path.join(INDICES_DIR, f"valid_mask_fold{k}.npy"))
        test_ds = MappedDataset(NPY_FILE, os.path.join(INDICES_DIR, f"test_mask_fold{k}.npy"))
        
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
        
        print(f" Sizes: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")
        
        # 2. Initialize Model
        model = EncoderMapNet(input_dim=INPUT_DIM, bottleneck_dim=BOTTLENECK_DIM).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        epoch_train_losses = []
        epoch_val_losses = []
        epoch_test_losses = []
        epoch_val_accs = []
        epoch_test_accs = []
        
        # ADDED TQDM FOR EPOCHS
        for epoch in tqdm(range(EPOCHS), desc=f"Fold {k} Epochs", leave=False):
            t_loss = train_epoch(model, train_loader, optimizer, device)
            v_loss, v_acc, _, _ = evaluate(model, val_loader, device)
            te_loss, te_acc, _, _ = evaluate(model, test_loader, device)
            
            epoch_train_losses.append(t_loss)
            epoch_val_losses.append(v_loss)
            epoch_test_losses.append(te_loss)
            epoch_val_accs.append(v_acc)
            epoch_test_accs.append(te_acc)
            
            print(f"  Epoch {epoch+1}/{EPOCHS} | Train Loss: {t_loss:.5f} | Val Loss: {v_loss:.5f} (Acc: {v_acc:.4f}) | Test Loss: {te_loss:.5f} (Acc: {te_acc:.4f})")
        
        # 4. Final Test Evaluation & Save Predictions
        final_test_loss, final_test_acc, final_cm, test_predictions = evaluate(model, test_loader, device)
        
        np.save(os.path.join(OUTPUT_DIR, f"test_predictions_fold{k}.npy"), test_predictions)
        
        # Save Confusion Matrix Plot
        plt.figure(figsize=(6, 5))
        sns.heatmap(final_cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Non-Contact', 'Contact'], 
                    yticklabels=['Non-Contact', 'Contact'])
        plt.title(f"Fold {k} Test Confusion Matrix (Threshold < {CONTACT_THRESHOLD})")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"confusion_matrix_fold{k}.png"))
        plt.close()
        
        # Save Loss/Accuracy Curves
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(epoch_train_losses, label='Train Loss')
        plt.plot(epoch_val_losses, label='Val Loss')
        plt.plot(epoch_test_losses, label='Test Loss')
        plt.title(f"Fold {k} Loss Curve")
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(epoch_val_accs, label='Val Accuracy')
        plt.plot(epoch_test_accs, label='Test Accuracy')
        plt.title(f"Fold {k} Accuracy Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"curves_fold{k}.png"))
        plt.close()
        
        fold_metrics['train_loss'].append(epoch_train_losses[-1])
        fold_metrics['val_loss'].append(final_test_loss) # Using final test loss for summary
        fold_metrics['test_loss'].append(final_test_loss)
        fold_metrics['val_acc'].append(epoch_val_accs[-1])
        fold_metrics['test_acc'].append(final_test_acc)
        
        print(f"✅ Fold {k} Complete. Final Test Acc: {final_test_acc:.4f}")
        
        # Clear GPU memory between folds
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 5. Print Final Summary
    print("\n" + "="*50)
    print(" 10-FOLD CROSS VALIDATION COMPLETE")
    print("="*50)
    print(f"Mean Test Accuracy: {np.mean(fold_metrics['test_acc']):.4f} ± {np.std(fold_metrics['test_acc']):.4f}")
    print(f"Mean Test Loss:     {np.mean(fold_metrics['test_loss']):.5f} ± {np.std(fold_metrics['test_loss']):.5f}")
    print(f"Results saved in: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()