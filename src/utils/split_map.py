import pandas as pd
from glob import glob
from utils.chem import get_scaffold

def create_split_map(data_path, data_file_prefix, split_map_name):
    files = glob(f"{data_path}/{data_file_prefix}*.parquet")
    rows = []

    for f in files:
        df = pd.read_parquet(f, columns=["activity_id", "smiles"])
        df["scaffold"] = df.smiles.map(get_scaffold)
        rows.extend(df[["activity_id", "scaffold"]].dropna().values.tolist())

    scaffold_df = pd.DataFrame(rows, columns=["activity_id", "scaffold"])
    scaffold_sizes = scaffold_df.groupby("scaffold").size().sort_values(ascending=False)

    train_frac, val_frac = 0.8, 0.1
    n_total = len(scaffold_df)
    train_cutoff = train_frac * n_total
    val_cutoff = (train_frac + val_frac) * n_total

    scaffold_to_split = {}
    count = 0

    for scaffold, size in scaffold_sizes.items():
        if count < train_cutoff:
            scaffold_to_split[scaffold] = "train"
        elif count < val_cutoff:
            scaffold_to_split[scaffold] = "val"
        else:
            scaffold_to_split[scaffold] = "test"
        count += size

    scaffold_df["split"] = scaffold_df.scaffold.map(scaffold_to_split)
    scaffold_df[["activity_id", "split"]].to_parquet(f"{data_path}/{split_map_name}", index=False)