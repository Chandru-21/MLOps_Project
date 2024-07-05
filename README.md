# Machine Learning Opearations (MLOps)

## MLOps maturity level 4

### Architecture

![image](https://github.com/Chandru-21/MLOps_Project/assets/64595758/123511be-fe66-424d-8776-513b908840fe)

### Data Versioning : DVC

### Continuous Integration(CI) : Triggered through ‘main.yml’ , building the code (docker), tests the code(Pytest),pushes the docker image to AWS ECR. 

### Experiment Tracking / Model Versioning : MLflow 

### Continuous Deployment(CD) : Deploys FastAPI in AWS EKS(kubernetes cluster) for real-time and batch predictions. 

### Continuous Monitoring(CM) : Integrating the ‘/metrics’ method of  FastAPI in Prometheus and visualizes endpoints in Grafana.  

### Continuous Training(CT) : Triggers code execution through GitHub Actions when new data is pushed to the remote DVC location and committed to Git. 

### Drift Monitoring : Uses a Streamlit app to monitor data drift, target drift, and data summary.

