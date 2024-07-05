# Machine Learning Opearations (MLOps)

## MLOps maturity level 4

## Overview :
This project implements a robust MLOps pipeline, facilitating the continuous integration, continuous deployment, and monitoring of machine learning models. The infrastructure leverages AWS, Kubernetes, and various open-source tools to ensure scalability, reproducibility, and maintainability.



## Architecture :

![image](https://github.com/Chandru-21/MLOps_Project/assets/64595758/123511be-fe66-424d-8776-513b908840fe)

## Key Features :

**Data Versioning** : DVC

**Continuous Integration(CI)** : Triggered through ‘main.yml’ , building the code (docker), tests the code(Pytest),pushes the docker image to AWS ECR. 

**Experiment Tracking / Model Versioning** : MLflow 

**Continuous Deployment(CD)** : Deploys FastAPI in AWS EKS(kubernetes cluster) for real-time and batch predictions. 

**Continuous Monitoring(CM)** : Integrating the ‘/metrics’ method of  FastAPI in Prometheus and visualizes endpoints in Grafana.  

**Continuous Training(CT)** : Triggers code execution through GitHub Actions when new data is pushed to the remote DVC location and committed to Git. 

**Drift Monitoring** : Uses a Streamlit app to monitor data drift, target drift, and performs data quality checks.






## Data Monitoring :

**Data Drift Monitoring** :

![image](https://github.com/Chandru-21/MLOps_Project/assets/64595758/af0df23d-9980-4ee4-94c0-ddebdb923237)

**Data Quality checks** :

![image](https://github.com/Chandru-21/MLOps_Project/assets/64595758/c1c62d64-9b69-4ca7-ba45-45ae226a7620)


## Continuous Monitoring(CM)

**Exposing "/metrics" on FastAPI to be connected to Prometheus** :

![fastapi](https://github.com/Chandru-21/MLOps_Project/assets/64595758/09b18b44-8cb1-4a86-9172-c79082cb77c8)

**FastAPI integrated in to Prometheus** :

![fastapi_prometheus](https://github.com/Chandru-21/MLOps_Project/assets/64595758/4b21c089-bef3-4e39-b5e1-04cb8e026345)

**Integrating Prometheus in to Grafana for Visualization** :

Monitoring FastAPI methods on Grafana,

![fastapi_continuous_monitoring](https://github.com/Chandru-21/MLOps_Project/assets/64595758/930f0a9a-352f-41f9-8106-9b6735af8ce4)

**Monitoring the resources of the Kubernetes cluster on Grafana** :

![grafana_monitoring_containers_dashboard](https://github.com/Chandru-21/MLOps_Project/assets/64595758/d046d9f9-1477-4975-9041-f4aa128bb0f3)











