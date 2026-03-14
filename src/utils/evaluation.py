import numpy as np
import pandas as pd
import torch
import gc
from sklearn.metrics import mean_squared_error, r2_score


def evaluate_model(model, data_files_location, splits_location):
    model.eval()
    splits_df = pd.read_parquet(splits_location)

    all_test_preds = []
    all_test_true = []

    with torch.no_grad():
        for file_path in data_files_location:
            df_batch = pd.read_parquet(file_path)
            df_batch = pd.merge(df_batch, splits_df, on="activity_id", how="inner")

            df_test = df_batch[df_batch["split"] == "test"]

            if len(df_test) == 0:
                continue

            X_test_np = np.stack(df_test["morgan_fp"].values).astype(np.float32)
            y_test = df_test["pic50"].values

            X_test_t = torch.from_numpy(X_test_np)
            y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

            test_preds = model(X_test_t)

            all_test_preds.extend(test_preds.flatten().tolist())
            all_test_true.extend(y_test_t.flatten().tolist())

            if "X_test_np" in locals():
                del X_test_np, X_test_t, y_test, y_test_t
            
            del df_batch, df_test
            gc.collect()

    test_r2 = r2_score(all_test_true, all_test_preds)
    print(f"\nTest split of ({len(all_test_true)} molecules):")
    print(f"R2: {test_r2:.3f}")
    return test_r2
