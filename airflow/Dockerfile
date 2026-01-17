FROM apache/airflow:3.1.0

USER root

RUN wget -O /tmp/jdk-21.deb https://download.oracle.com/java/21/latest/jdk-21_linux-x64_bin.deb

RUN apt-get update && apt-get install -y /tmp/jdk-21.deb && rm /tmp/jdk-21.deb

RUN mkdir -p /opt/airflow/data && chown -R airflow: /opt/airflow/data
USER airflow

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt