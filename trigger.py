import logging
import time
from datetime import datetime, timedelta
from azure.storage.blob import BlobServiceClient
from retraining import get_best_model, update_registry, retrain_model

def check_for_new_blobs(blob_service_client, container_client, container_name, last_checked_blob_name=None):
    new_blobs = []
    blobs = container_client.list_blobs()  
    sorted_blobs = sorted(blobs, key=lambda blob: blob.last_modified, reverse=True)
    recent_blobs = sorted_blobs[:1]
    for blob in recent_blobs:
        if blob.name.endswith("0"):
            if last_checked_blob_name is None or blob.name > last_checked_blob_name:
                new_blobs.append(blob.name)

    return new_blobs

def trigger_retraining(blob_names):
    # Replace this with your actual logic for downloading images and triggering retraining using MLflow
    logging.info(f"Found {len(blob_names)} new images: {', '.join(blob_names)}")
    # Implement logic to download the new images using the blob names
    # ... (download logic using blob_service_client)...
    retrain_model()
    # Trigger retraining using downloaded images and MLflow
    # ... (MLflow training logic)...

if __name__ == "__main__":
    logging.basicConfig(filename="retraining.log", level=logging.INFO)
    # Azure Blob Storage connection details 
    container_name = "images"
    # Set up Blob service client
    connection_string = ""
    blob_service_client = BlobServiceClient.from_connection_string(conn_str=connection_string)
    container_client = blob_service_client.get_container_client("images")
    

    # Variable to store the last checked blob name (optional for incremental checking)
    last_checked_blob_name = None

    while True:
        # Check for new blobs every minute
        new_blobs = check_for_new_blobs(blob_service_client, container_client, container_name, last_checked_blob_name)

        if new_blobs:
            # Trigger retraining if new blobs are found
            trigger_retraining(new_blobs)

            # Update last checked blob name for incremental check (optional)
            last_checked_blob_name = new_blobs[-1]  # Use the last new blob name

        logging.info("Sleeping for 1 minute...")
        time.sleep(60)  # Sleep for 60 seconds before checking again
