import pandas as pd
import numpy as np
from prediction_model.config import config  
import mlflow



def generate_predictions(data_input):
    data = pd.DataFrame(data_input)
    experiment_name = config.EXPERIMENT_NAME
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id
    runs_df=mlflow.search_runs(experiment_ids=experiment_id,order_by=['metrics.f1_score DESC'])
    best_run=runs_df.iloc[0]
    best_run_id=best_run['run_id']
    best_model='runs:/' + best_run_id + config.MODEL_NAME
    loan_prediction_model=mlflow.sklearn.load_model(best_model)
    prediction=loan_prediction_model.predict(data)
    output = np.where(prediction==1,'Y','N')
    result = {"prediction":output}
    return result


def generate_predictions_batch(data_input):
    # data = pd.DataFrame(data_input)
    experiment_name = config.EXPERIMENT_NAME
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id
    runs_df=mlflow.search_runs(experiment_ids=experiment_id,order_by=['metrics.f1_score DESC'])
    best_run=runs_df.iloc[0]
    best_run_id=best_run['run_id']
    best_model='runs:/' + best_run_id + config.MODEL_NAME
    loan_prediction_model=mlflow.sklearn.load_model(best_model)
    prediction=loan_prediction_model.predict(data_input)
    output = np.where(prediction==1,'Y','N')
    result = {"prediction":output}
    return result


    


if __name__=='__main__':
    generate_predictions()