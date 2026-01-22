from airflow.sdk import dag, task
from src.preprocess_data import get_and_preprocess_data


@dag(dag_id="prepare_chembl_data", schedule="@once")
def prepare_raw_data():
    @task
    def prepare_data_chunks():
        get_and_preprocess_data()

    prepare_data_chunks()


raw_data_dag = prepare_raw_data()
