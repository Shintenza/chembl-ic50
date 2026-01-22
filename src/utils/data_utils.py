import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors, QED

from sklearn.preprocessing import OneHotEncoder


def normalize_to_nM(
    df, value_col="ic50", units_col="standard_units", mw_col="mw_freebase"
):
    ug_mask = df[units_col] == "ug.mL-1"
    df.loc[ug_mask, value_col] = (
        df.loc[ug_mask, value_col] / df.loc[ug_mask, mw_col]
    ) * 1e6
    df.loc[ug_mask, units_col] = "nM"


def compute_pic50(df, value_col="ic50"):
    df["pic50"] = -np.log10(df[value_col] * 1e-9)


def drop_pic50_outliers(df, value_col="pic50", lower_bound=3, upper_bound=11):
    df = df[(df[value_col] >= lower_bound) & (df[value_col] <= upper_bound)]


def impute_properties_from_smiles(df, smiles_col="smiles"):
    for i, row in df.iterrows():
        smi = row[smiles_col]
        if pd.isna(smi):
            continue

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        # MW_FREEBASE
        if pd.isna(row.get("mw_freebase")):
            df.at[i, "mw_freebase"] = Descriptors.MolWt(mol)  # type: ignore

        # ALOGP
        if pd.isna(row.get("alogp")):
            df.at[i, "alogp"] = Descriptors.MolLogP(mol)  # type: ignore

        # HBA / HBD
        if pd.isna(row.get("hba")):
            df.at[i, "hba"] = Lipinski.NumHAcceptors(mol)  # type: ignore
        if pd.isna(row.get("hbd")):
            df.at[i, "hbd"] = Lipinski.NumHDonors(mol)  # type: ignore

        # PSA
        if pd.isna(row.get("psa")):
            df.at[i, "psa"] = rdMolDescriptors.CalcTPSA(mol)

        # RTB
        if pd.isna(row.get("rtb")):
            df.at[i, "rtb"] = Lipinski.NumRotatableBonds(mol)  # type: ignore

        # RO3_PASS
        if pd.isna(row.get("ro3_pass")):
            mw = Descriptors.MolWt(mol)  # type: ignore
            logp = Descriptors.MolLogP(mol)  # type: ignore
            hba = Lipinski.NumHAcceptors(mol)  # type: ignore
            hbd = Lipinski.NumHDonors(mol)  # type: ignore
            ro3 = (mw < 300) and (logp < 3) and (hba <= 3) and (hbd <= 3)
            df.at[i, "ro3_pass"] = "Y" if ro3 else "N"

        # AROMATIC_RINGS
        if pd.isna(row.get("aromatic_rings")):
            df.at[i, "aromatic_rings"] = Lipinski.NumAromaticRings(mol)  # type: ignore

        # HEAVY_ATOMS
        if pd.isna(row.get("heavy_atoms")):
            df.at[i, "heavy_atoms"] = rdMolDescriptors.CalcNumHeavyAtoms(mol)

        # QED_WEIGHTED
        if pd.isna(row.get("qed_weighted")):
            df.at[i, "qed_weighted"] = QED.qed(mol)

        # NUM_RO5_VIOLATIONS
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


def one_hot_encode_assay_type(df):
    ASSAY_TYPES = ["A", "B", "F", "P", "T", "U"]

    ohe = OneHotEncoder(
        categories=[ASSAY_TYPES], handle_unknown="ignore", sparse_output=False
    )

    encoded_array = ohe.fit_transform(df[["assay_type"]])
    encoded_cols = ohe.get_feature_names_out(["assay_type"])
    encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols, index=df.index)
    df.drop("assay_type", axis=1, inplace=True)
    df[encoded_cols] = encoded_df


def drop_columns(df):
    COLUMNS_TO_DROP = [
        "activity_id",
        "assay_id",
        "standard_units",
        "full_molformula",
        "target_id",
    ]

    duplicated_cols = df.columns[df.columns.duplicated(keep="first")]
    if len(duplicated_cols) > 0:
        df.drop(columns=duplicated_cols, inplace=True)

    cols_to_drop = [c for c in COLUMNS_TO_DROP if c in df.columns]
    if len(cols_to_drop) > 0:
        df.drop(columns=cols_to_drop, inplace=True)


def process_single_data_chunk(df: pd.DataFrame) -> pd.DataFrame:
    normalize_to_nM(df)
    compute_pic50(df)
    drop_pic50_outliers(df)
    impute_properties_from_smiles(df)
    one_hot_encode_assay_type(df)
    drop_columns(df)

    return df
