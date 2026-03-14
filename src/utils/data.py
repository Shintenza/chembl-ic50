import pandas as pd
from utils.chem import get_fingerprint

def compute_fingerprint_column(df: pd.DataFrame, smiles_col_name = 'smiles', fp_col_name="morgan_fp"):
    df[fp_col_name] = df[smiles_col_name].apply(get_fingerprint)