import os
import shutil
import numpy as np
from azure.storage.blob import BlobServiceClient
from PIL import Image
from keras.utils import to_categorical

def download_images():
    connection_string = ""
    blob_service_client = BlobServiceClient.from_connection_string(conn_str=connection_string)
    container_client = blob_service_client.get_container_client("images")
    blobs = container_client.list_blobs()
    sorted_blobs = sorted(blobs, key=lambda blob: blob.last_modified, reverse=True)
    num_blobs = min(10,len(sorted_blobs))
    recent_blobs = sorted_blobs[:num_blobs]
    local_dir = "local_retraining_mlflow\\artifacts\images"
    if os.path.exists(local_dir):
        shutil.rmtree(local_dir)
    else :
        print("Directory does not exist")
    os.makedirs(local_dir)
    labels = []
    x_train = []
    
    for blob in recent_blobs:
        blob_client = blob_service_client.get_blob_client("images", blob.name)
        download_file_path = os.path.join(local_dir, blob.name+'.png')
        with open(download_file_path, "wb") as download_file:
            download_file.write(blob_client.download_blob().readall())
        with Image.open(download_file_path) as img:
            img = img.resize((28, 28))  # Resize the image
            img = img.convert('L')  # Convert the image to grayscale
            img_array = np.array(img)
            x_train.append(img_array)
    labels = [int(blob.name.split("_")[0]) for blob in recent_blobs]
    ytrain = to_categorical(labels, num_classes=10)
    xtrain = np.array(x_train)
    return xtrain, ytrain

def main():
    return download_images()

if __name__ == "__main__":
    main()


