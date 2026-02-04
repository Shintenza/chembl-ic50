from src.utils.consts import CLEANED_DATA_LOCATION, CLEANED_DATA_SPLIT_MAP
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from glob import glob



def get_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)



def create_split_map():
    files = glob(f"{CLEANED_DATA_LOCATION}/raw_batch_*.parquet")
    rows = []

    for f in files:
        df = pd.read_parquet(f, columns=["activity_id", "smiles"])
        for rid, smi in zip(df.activity_id, df.smiles):
            scaffold = get_scaffold(smi)
            if scaffold:
                rows.append((rid, scaffold))

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
    scaffold_df[["activity_id", "split"]].to_parquet(CLEANED_DATA_SPLIT_MAP, index=False)
