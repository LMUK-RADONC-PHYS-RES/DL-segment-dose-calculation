import os
import sys
import argparse
import shutil
from datetime import datetime

import torch
import torch.nn as nn
from tqdm import tqdm
from monai.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "model", "CNN_Mamba"))

from utils import MultiModalHDF5Dataset
from model import CNN_Mamba2_L2_SpatialMix


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0.0
    for batch in dataloader:
        ct    = batch[0][0].to(device)
        proj  = batch[0][1].to(device)
        label = batch[1].to(device)

        optimizer.zero_grad()
        output = model(ct, proj)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)


def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            ct    = batch[0][0].to(device)
            proj  = batch[0][1].to(device)
            label = batch[1].to(device)

            output = model(ct, proj)
            loss = criterion(output, label)
            epoch_loss += loss.item()
    return epoch_loss / len(dataloader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CNN-Mamba dose prediction model')
    parser.add_argument('--data-dir', required=True,
                        help='Directory containing ct_dataset.h5, proj_dataset.h5, dose_dataset.h5')
    parser.add_argument('--out-dir', default=None,
                        help='Output directory (default: ./results/<timestamp>_Mamba)')
    parser.add_argument('--device', default='cuda:0',
                        help='PyTorch device string (default: cuda:0)')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr-patience', type=int, default=15)
    parser.add_argument('--num-workers', type=int, default=4)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model  = CNN_Mamba2_L2_SpatialMix().to(device)

    train_ds = MultiModalHDF5Dataset(
        ct_hdf_path=os.path.join(args.data_dir, 'ct_dataset.h5'),
        proj_hdf_path=os.path.join(args.data_dir, 'proj_dataset.h5'),
        dose_hdf_path=os.path.join(args.data_dir, 'dose_dataset.h5'),
        group_name='train',
    )
    val_ds = MultiModalHDF5Dataset(
        ct_hdf_path=os.path.join(args.data_dir, 'ct_dataset.h5'),
        proj_hdf_path=os.path.join(args.data_dir, 'proj_dataset.h5'),
        dose_hdf_path=os.path.join(args.data_dir, 'dose_dataset.h5'),
        group_name='validation',
    )

    current_time  = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    experiment_dir = args.out_dir or os.path.join('results', f'{current_time}_Mamba')
    log_dir    = os.path.join(experiment_dir, 'log')
    model_dir  = os.path.join(experiment_dir, 'model')
    script_dir = os.path.join(experiment_dir, 'script')
    os.makedirs(log_dir,    exist_ok=True)
    os.makedirs(model_dir,  exist_ok=True)
    os.makedirs(script_dir, exist_ok=True)
    shutil.copy(os.path.abspath(__file__), os.path.join(script_dir, os.path.basename(__file__)))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    print(f"Device: {device}")
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                  patience=args.lr_patience, verbose=True)

    best_val_loss = float('inf')
    log_file = os.path.join(log_dir, 'run.log')
    with open(log_file, 'w') as f:
        for epoch in tqdm(range(args.epochs), desc="Training Progress"):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss   = validate_one_epoch(model, val_loader, criterion, device)
            current_lr = optimizer.param_groups[0]['lr']
            msg = (f"Epoch {epoch+1}/{args.epochs}, LR: {current_lr:.2e}, "
                   f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            tqdm.write(msg)
            f.write(msg + '\n')

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(model_dir, 'best_model.pth'))
                msg = "Model saved"
                tqdm.write(msg)
                f.write(msg + '\n')
            f.flush()
