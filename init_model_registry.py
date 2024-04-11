import mlflow
import mlflow.keras
from keras.models import load_model
mlflow.set_tracking_uri("http://localhost:8080")
# experiment_id = mlflow.create_experiment("Exp_1")
# mlflow.set_experiment(experiment_id)
experiment = mlflow.get_experiment_by_name("Exp_1").experiment_id
if experiment == None:
    experiment_id = mlflow.create_experiment("Exp_1")
else :
    experiment_id = mlflow.get_experiment_by_name("Exp_1").experiment_id

mlflow.set_experiment(experiment_id)

model = load_model("new_mnist(2x2)_epoch8_compile_changes_optimum.h5")
if(model is None):
    print("Model not loaded")
else:
    print("Model loaded successfully")
with mlflow.start_run(experiment_id=experiment_id):
    mlflow.keras.log_model(model, "model_1")
    run_id = mlflow.active_run().info.run_id
    mlflow.log_metric("accuracy", 0.9825)
    mlflow.log_metric("loss", 0.0396)
    mlflow.log_metric("val_accuracy",0.9915)
    mlflow.log_metric("val_loss",0.0235)

print(f"Run ID: {run_id}")
print(f"Artifact URI: mlruns/{experiment_id}/{run_id}/artifacts/model_1")

#Register the model
model_uri = f"runs:/{run_id}/model_1"
model_details = mlflow.register_model(model_uri, "MNIST")
