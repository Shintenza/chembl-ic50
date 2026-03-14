import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gc
from torch.utils.data import TensorDataset, DataLoader


def train_baseline_mlp(
    data_files_location,
    splits_location,
    model,
    criterion=nn.MSELoss,
    optimizer=optim.Adam,
    lr=0.001,
    epochs=25,
    batch_size=128,
):
    splits_df = pd.read_parquet(splits_location)

    loss_fn = criterion()
    optim = optimizer(model.parameters, lr)

    for epoch in range(epochs):
        epoch_train_loss = 0.0
        train_samples_count = 0

        epoch_val_loss = 0.0
        val_samples_count = 0

        for file_path in data_files_location:
            df_batch = pd.read_parquet(file_path)
            df_batch = pd.merge(df_batch, splits_df, on="activity_id", how="inner")

            df_train = df_batch[df_batch["split"] == "train"]
            df_val = df_batch[df_batch["split"] == "val"]

            if len(df_train) == 0:
                continue

            X_np = np.stack(df_train["morgan_fp"].values).astype(np.float32)
            y_train = df_train["pic50"].values

            X_train_t = torch.from_numpy(X_np)
            y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

            loader = DataLoader(
                TensorDataset(X_train_t, y_train_t),
                batch_size=batch_size,
                shuffle=True,
            )

            model.train()
            for bX, by in loader:
                optim.zero_grad()
                preds = model(bX)
                loss = loss_fn(preds, by)
                loss.backward()
                optim.step()

                epoch_train_loss += loss.item() * len(bX)
                train_samples_count += len(bX)

            if len(df_val) > 0:
                X_val_np = np.stack(df_val["morgan_fp"].values).astype(np.float32)
                y_val = df_val["pic50"].values

                X_val_t = torch.from_numpy(X_val_np)
                y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

                model.eval()

                with torch.no_grad():
                    val_preds = model(X_val_t)
                    v_loss = loss_fn(val_preds, y_val_t)

                    epoch_val_loss += v_loss.item() * len(X_val_t)
                    val_samples_count += len(X_val_t)

                del df_batch, df_train
                if "X_train_t" in locals():
                    del X_np, X_train_t, y_train, y_train_t, loader
                if "X_val_np" in locals():
                    del X_val_np, X_val_t, y_val, y_val_t
                gc.collect()

                print("single batch finished")

        avg_train_loss = (
            epoch_train_loss / train_samples_count if train_samples_count > 0 else 0
        )

        avg_val_loss = (
            epoch_val_loss / val_samples_count if val_samples_count > 0 else 0
        )
        print(
            f"Epoch [{epoch+1:3d}/{epochs}] | Train MSE: {avg_train_loss:.4f} | Val MSE: {avg_val_loss:.4f}"
        )
