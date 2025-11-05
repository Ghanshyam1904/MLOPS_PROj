#!/bin/bash
nohup airflow scheduler &
airflow webserver
