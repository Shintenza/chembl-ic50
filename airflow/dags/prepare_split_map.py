from airflow.sdk import dag, task
from src.prepare_split_map import create_split_map


@dag(dag_id="prepare_split_map", schedule="@once")
def prepare_split_map():
    @task
    def generate_split_map():
        create_split_map()
        

    generate_split_map()


raw_data_dag = prepare_split_map()
