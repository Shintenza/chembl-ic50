import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors, QED

def normalize_to_nM(df, value_col="ic50", units_col="standard_units", mw_col="mw_freebase"):
    supported_units = ['nM', 'uM', 'µM', 'mM', 'pM', '10^2 uM', '10^-5 uM',
                       'ug.mL-1', 'ug']
    
    df = df[df[units_col].isin(supported_units)].copy()
    
    def convert_row(row):
        value = row[value_col]
        unit = row[units_col]
        mw = row.get(mw_col, None)
        
        if pd.isna(value) or pd.isna(unit):
            return np.nan
        
        if unit == 'nM':
            return value
        elif unit in ['uM', 'µM']:
            return value * 1e3
        elif unit == 'mM':
            return value * 1e6
        elif unit == 'pM':
            return value * 1e-3
        elif unit == '10^2 uM':
            return value * 1e2 * 1e3
        elif unit == '10^-5 uM':
            return value * 1e-5 * 1e3
        elif unit in ['ug.mL-1', 'ug']:
            if mw is None or mw == 0:
                return np.nan
            return (value / mw) * 1e6
        else:
            return np.nan
    
    df[value_col] = df.apply(convert_row, axis=1)
    df.dropna(subset=[value_col], inplace=True)


def compute_pic50(df, value_col="ic50"):
    df["pic50"] = -np.log10(df[value_col] * 1e-9)


def drop_pic50_outliers(df, value_col="pic50", lower_bound=3, upper_bound=12):
    mask = (df[value_col] >= lower_bound) & (df[value_col] <= upper_bound)
    df.drop(df[~mask].index, inplace=True)


def impute_properties_from_smiles(df, smiles_col="smiles"):
    drop_indices = []
    for i, row in df.iterrows():
        smi = row[smiles_col]
        if pd.isna(smi):
            drop_indices.append(i)
            continue

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            drop_indices.append(i)
            continue

        if pd.isna(row.get("mw_freebase")):
            df.at[i, "mw_freebase"] = Descriptors.MolWt(mol)  # type: ignore

        if pd.isna(row.get("alogp")):
            df.at[i, "alogp"] = Descriptors.MolLogP(mol)  # type: ignore

        if pd.isna(row.get("hba")):
            df.at[i, "hba"] = Lipinski.NumHAcceptors(mol)  # type: ignore
        if pd.isna(row.get("hbd")):
            df.at[i, "hbd"] = Lipinski.NumHDonors(mol)  # type: ignore

        if pd.isna(row.get("psa")):
            df.at[i, "psa"] = rdMolDescriptors.CalcTPSA(mol)

        if pd.isna(row.get("rtb")):
            df.at[i, "rtb"] = Lipinski.NumRotatableBonds(mol)  # type: ignore

        if pd.isna(row.get("ro3_pass")):
            mw = Descriptors.MolWt(mol)  # type: ignore
            logp = Descriptors.MolLogP(mol)  # type: ignore
            hba = Lipinski.NumHAcceptors(mol)  # type: ignore
            hbd = Lipinski.NumHDonors(mol)  # type: ignore
            ro3 = (mw < 300) and (logp < 3) and (hba <= 3) and (hbd <= 3)
            df.at[i, "ro3_pass"] = "Y" if ro3 else "N"

        if pd.isna(row.get("aromatic_rings")):
            df.at[i, "aromatic_rings"] = Lipinski.NumAromaticRings(mol)  # type: ignore

        if pd.isna(row.get("heavy_atoms")):
            df.at[i, "heavy_atoms"] = rdMolDescriptors.CalcNumHeavyAtoms(mol)

        if pd.isna(row.get("qed_weighted")):
            df.at[i, "qed_weighted"] = QED.qed(mol)

        if pd.isna(row.get("num_ro5_violations")):
            violations = 0
            if Descriptors.MolWt(mol) > 500:  # type: ignore
                violations += 1
            if Descriptors.MolLogP(mol) > 5:  # type: ignore
                violations += 1
            if Lipinski.NumHAcceptors(mol) > 10:  # type: ignore
                violations += 1
            if Lipinski.NumHDonors(mol) > 5:  # type: ignore
                violations += 1
            df.at[i, "num_ro5_violations"] = violations
    df.drop(index=drop_indices, inplace=True)

def yn_to_binary(df, column):
    mapping = {"Y": 1, "N": 0}
    df[column].replace(mapping, inplace=True)

def drop_columns(df):
    COLUMNS_TO_DROP = [
        "assay_id",
        "standard_units",
        "full_molformula",
        "np_likness_score"
    ]

    duplicated_cols = df.columns[df.columns.duplicated(keep="first")]
    if len(duplicated_cols) > 0:
        df.drop(columns=duplicated_cols, inplace=True)

    cols_to_drop = [c for c in COLUMNS_TO_DROP if c in df.columns]
    if len(cols_to_drop) > 0:
        df.drop(columns=cols_to_drop, inplace=True)


def process_single_data_chunk(df: pd.DataFrame) -> pd.DataFrame:
    impute_properties_from_smiles(df)
    normalize_to_nM(df)
    compute_pic50(df)
    drop_pic50_outliers(df)
    drop_columns(df)
    yn_to_binary(df, 'ro3_pass')

    return df