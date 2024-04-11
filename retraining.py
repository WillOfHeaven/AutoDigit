import mlflow
from mlflow.tracking import MlflowClient
from data import download_images
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense,Flatten
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras import backend as K
import matplotlib.pyplot as plt
import cv2
from keras.utils import to_categorical
from mlflow.exceptions import MlflowException

def get_best_model(model_name):
    # Set the tracking URI
    mlflow.set_tracking_uri("http://localhost:8080")
    # Initialize the MLflow client
    client = MlflowClient()

    # Get all versions of the model
    model_versions = client.search_model_versions(f"name='{model_name}'")

    # Find the model version with the highest accuracy
    best_version = None
    best_accuracy = -1
    for mv in model_versions:
        run = client.get_run(mv.run_id)
        accuracy = run.data.metrics["val_accuracy"]  # Replace "accuracy" with the name of your accuracy metric
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_version = mv.version

    # Load the best model
    model_uri = f"models:/{model_name}/{best_version}"
    print("best model uri")
    print (model_uri)
    print (best_accuracy)
    physical_uri = client.get_model_version_download_uri(model_name, best_version)
    print("Physical uri ",physical_uri)
    model = mlflow.keras.load_model(model_uri)
    return model,model_uri

def update_registry(model_name, model, accuracy,loss,val_loss,val_accuracy):
    mlflow.set_tracking_uri("http://localhost:8080")
    client = MlflowClient()
    
    # Check if the model exists in the registry
    try:
        client.get_registered_model(model_name)
    except MlflowException:
        # If not, create it
        client.create_registered_model(model_name)
  
    experiment = mlflow.get_experiment_by_name("Exp_1").experiment_id
    if experiment == None:
        experiment_id = mlflow.create_experiment("Exp_1")
    else :
        print("Experiment exists")
        experiment_id = mlflow.get_experiment_by_name("Exp_1").experiment_id

    mlflow.set_experiment("Exp_1")

    with mlflow.start_run(experiment_id=experiment_id):
        mlflow.keras.log_model(model, "model")
        run_id = mlflow.active_run().info.run_id
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("loss", loss)
        mlflow.log_metric("val_loss",val_loss)
        mlflow.log_metric("val_accuracy",val_accuracy)
    print(f"Run ID: {run_id}")
    print(f"Artifact URI: mlruns/{experiment_id}/{run_id}/artifacts/model_1")

    #Register the model
    client.create_model_version(model_name, f"runs:/{run_id}/model", run_id)

def retrain_model():
    # Load the best model
    model,model_uri = get_best_model("MNIST")
    print("Model loaded successfully",model_uri)
    # Retrain the model with new data
    
    # Data Preparation
    x_train, y_train = download_images()
    (xtrain,ytrain), (x_test,y_test) = mnist.load_data()
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    x_train = np.array([cv2.resize(img, (28, 28)) for img in x_train])
    x_test = np.array([cv2.resize(img, (28, 28)) for img in x_test])
    x_train = x_train.reshape(x_train.shape[0], 28,28,1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    # print(x_train.shape)
    # print(x_test.shape)
    y_test = to_categorical(y_test, num_classes=10)
    # print("The values of ytrain are ",y_train)
    # print("The values of ytrain are ",y_test)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # Model Definition
 # Retrain the model
    #batch_size = 128
    #num_classes = 10
    #epochs = 1
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test))

    # Get the loss, validation loss, accuracy, and validation accuracy
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    # Evaluate the model
    print(f"Loss: {loss}, Accuracy: {accuracy}")
    update_registry("MNIST",model, accuracy[-1],loss[-1],val_loss[-1],val_accuracy[-1])

def main():
    retrain_model()

if __name__ == "__main__":
    main()