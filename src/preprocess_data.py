from src.utils.raw_data_query import get_data_query
from src.utils.data_utils import process_single_data_chunk
from src.utils.consts import CLEANED_DATA_LOCATION, CHUNK_SIZE

from sqlalchemy import create_engine
import os
import pandas as pd


def get_and_preprocess_data():
    db_connection_link = os.environ.get("CHEMBL_SQL_ALCHEMY_CONN")
    if db_connection_link == None:
        raise KeyError("missing sql alchemy connection url")

    db_query = get_data_query()
    engine = create_engine(db_connection_link)

    os.makedirs(CLEANED_DATA_LOCATION, exist_ok=True)

    with engine.connect() as conn:
        batch_index = 0
        for chunk in pd.read_sql(db_query, conn, chunksize=CHUNK_SIZE):
            processed_chunk = process_single_data_chunk(chunk)
            processed_chunk.to_parquet(
                f"{CLEANED_DATA_LOCATION}/raw_batch_{batch_index}.parquet",
                index=False,
            )
            batch_index += 1
