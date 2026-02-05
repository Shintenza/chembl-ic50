from src.utils.consts import GLOBAL_FEATURES, CLEANED_DATA_SPLIT_MAP, CLEANED_DATA_LOCATION, GLOBAL_FEATURES_SCALER
from sklearn.preprocessing import RobustScaler
import joblib
import pandas as pd
import glob


def prepare_global_features_scaller():
    split_map_df = pd.read_parquet(CLEANED_DATA_SPLIT_MAP)

    data_files = glob.glob(f"{CLEANED_DATA_LOCATION}/raw_batch_*.parquet")

    dfs = [pd.read_parquet(f, columns=GLOBAL_FEATURES + ['activity_id']) for f in data_files]
    data_df = pd.concat(dfs, ignore_index=True)
    data_df = data_df.merge(split_map_df[['activity_id', 'split']], on='activity_id', how='left')

    train_df = data_df.loc[data_df['split'] == 'train', GLOBAL_FEATURES]

    scaler = RobustScaler()
    scaler.fit(train_df)

    joblib.dump(scaler, GLOBAL_FEATURES_SCALER)