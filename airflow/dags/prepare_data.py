from websockets.legacy.framing import prepare_data
from sqlalchemy import create_engine
import os
from airflow.sdk import BaseOperator, dag, task
from utils.raw_data_query import get_data_query
from utils.data_utils import process_single_data_chunk

import pandas as pd


CHUNK_SIZE = 300000
CLEANED_DATA_LOCATION = "/opt/airflow/data/preprocessed"

@dag(
    dag_id="prepare_chembl_data",
    schedule="@once"
)
def prepare_raw_data():
    @task
    def prepare_data_directory():
        os.makedirs(CLEANED_DATA_LOCATION, exist_ok=True) 
    
    @task
    def prepare_data_chunks():
        db_connection_link = os.environ.get('CHEMBL_SQL_ALCHEMY_CONN')
        if db_connection_link == None:
            raise KeyError('missing sql alchemy connection url')
        db_query = get_data_query()
        engine = create_engine(db_connection_link)

        with engine.connect() as conn:
            batch_index = 0
            for chunk in pd.read_sql(db_query, conn, chunksize=CHUNK_SIZE):
                processed_chunk = process_single_data_chunk(chunk)
                processed_chunk.to_parquet(
                    f"{CLEANED_DATA_LOCATION}/raw_batch_{batch_index}.parquet", index=False
                )
                batch_index += 1

    prepare_data_directory()
    prepare_data_chunks()

raw_data_dag = prepare_raw_data()
