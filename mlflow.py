import mlflow

mlflow.set_tracking_uri("http://172.16.110.87:4000")
mlflow.set_experiment("DDRNET")
with mlflow.start_run(run_id = 'a8f7164bc822479c9ff4844cdaa7c52b'):

    mlflow.log_artifact('','')

