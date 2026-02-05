from airflow.sdk import dag, task
from src.prepare_global_features_scaller import prepare_global_features_scaller
from src.utils.consts import GLOBAL_FEATURES_SCALER

import os


@dag(dag_id="prepare_global_features_scaller", schedule="@once")
def prepare():
    @task
    def prepare_dir():
        os.makedirs(os.path.dirname(GLOBAL_FEATURES_SCALER), exist_ok=True)
    @task
    def fit_scaller():
        prepare_global_features_scaller()
        
    prepare_dir()
    fit_scaller()


raw_data_dag = prepare()
