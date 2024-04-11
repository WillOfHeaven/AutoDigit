import numpy as np
import cv2
import matplotlib.pyplot as plt
import keras
from keras.models import load_model
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from azure.storage.blob import BlobServiceClient
import datetime
import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import keras
from mlflow.exceptions import MlflowException

@st.cache_resource()
def init_azure_connection():
    connection_string = "Insert your connnection string here"
    blob_service_client = BlobServiceClient.from_connection_string(conn_str=connection_string)
    container_client = blob_service_client.get_container_client("images")
    return container_client


@st.cache_data
def get_today_date():
  today = datetime.date.today()
  return today.strftime("%Y-%m-%d")


def set_blob_tags(blob_client):
    tags = blob_client.get_blob_tags()
    updated_tags = {'Date': get_today_date()}
    tags.update(updated_tags)

def get_blob_cnt(container_client):
    blob_iterable = container_client.list_blobs()
    blob_count = sum(1 for _ in blob_iterable)
    return blob_count+1

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

def upload(img,cnt,correct_response):
    container_client =  init_azure_connection()
    # now = datetime.datetime.now()
    # print(now)
    # dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    blob_name = f"{correct_response}_{cnt}"
    blob_client = container_client.get_blob_client(blob_name)
    # Convert the image to bytes and upload it
    img_bytes = cv2.imencode('.png', img)[1].tobytes()
    blob_client.upload_blob(img_bytes)
    blob_iterable = container_client.list_blobs()
    blob_count = sum(1 for _ in blob_iterable)
    if(blob_client.exists()):
        st.success("Uploaded Successfully ! ")
    else:  
        st.error("Failed to upload the image")
    st.write(f"Total number of unique new dataset created {blob_count}")

    
    
def main():
    if "button_id" not in st.session_state:
        st.session_state["button_id"] = ""
    if "color_to_label" not in st.session_state:
        st.session_state["color_to_label"] = {}
    PAGES = {
        "About": about_page,
        "Basic example": full_app,
    }
    init_azure_connection()
    page = st.sidebar.selectbox("Page:", options=list(PAGES.keys()))
    PAGES[page]()

    with st.sidebar:
        st.markdown("---")
        st.markdown(
            '<h6>Made in &nbsp<img src="https://streamlit.io/images/brand/streamlit-mark-color.png" alt="Streamlit logo" height="16">&nbsp </h6>',
            unsafe_allow_html=True,
        )


def about_page():
    st.title("About This App")

    st.markdown("This application is a number recognition tool designed to showcase the power of combining Streamlit with deep learning models.")

    st.subheader("What it does:")

    st.write("""
    - Allows users to draw a digit (0-9) on a canvas.
    - Utilizes a Convolutional Neural Network (CNN) model to analyze the drawn digit.
    - Predicts the most likely number represented by the drawing.
    """)

    st.subheader("Technical Details:")

    st.write("""
    - Frontend framework: Streamlit (https://docs.streamlit.io/)
    - Drawing canvas: streamlit-drawable-canvas library (https://github.com/andfanilo/streamlit-drawable-canvas)
    - Machine learning model: User-defined CNN model trained on the MNIST dataset (replace with relevant details)
    """)

    st.subheader("Developed by:")

    st.write("""
    - Gaurav Rawat
    - Karthik Sharma Dhulipati
    - Mohak Kumar Srivastava
    - Naman Jain
    - Sambuddha Chatterjee
             """)

    st.subheader("Feedback :")

    st.write("""
    We welcome your feedback and contributions to this application! 
    - Feel free to report any issues or suggest improvements on [github](https://github.com/WillOfHeaven/MinorGP) (if applicable). 
    """)

def full_app():
    cnt = get_blob_cnt(init_azure_connection()) 
    st.sidebar.header("Configuration")
    st.markdown(
        """
    Draw on the canvas, get the drawings back to Streamlit!
    """
    ) 
    # Specify canvas parameters in application
    drawing_mode = "freedraw"
    stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 25)
    stroke_color = st.sidebar.color_picker("Stroke color hex: ", "#FFFFFF")
    bg_color = st.sidebar.color_picker("Background color hex: ", "#000000")
    bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
    realtime_update = st.sidebar.checkbox("Update in realtime", True)

    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=stroke_width ,
        stroke_color=stroke_color ,
        background_color=bg_color ,
        background_image=Image.open(bg_image) if bg_image else None,
        update_streamlit=realtime_update,
        height = 480,
        width = 480,
        drawing_mode=drawing_mode,
        display_toolbar=st.sidebar.checkbox("Display toolbar", True),
        key="full_app",
    )

    #st.button("Clear canvas", key="clear_canvas")
    from retraining import get_best_model, update_registry, retrain_model
    from data import download_images
    model,model_uri = get_best_model("MNIST")
    st.write("Model loaded successfully",model_uri)
    if canvas_result.image_data is not None:
        img = cv2.cvtColor(canvas_result.image_data, cv2.COLOR_RGBA2GRAY)  # Convert the image to grayscale  # Convert the image to grayscale
        img_resized = cv2.resize(img, (28, 28))  # Resize the image to 28x28
        img_reshaped = img_resized.reshape(1, 28, 28, 1)  # Reshape the image to match the model's expected input shape
        img_reshaped = img_reshaped.astype("float32")
        st.image(img_resized,caption="Resized Image")
        img_reshaped /= 255.0  # Normalize the image if your model was trained on normalized images
        response = model.predict(img_reshaped)  # Use the model to predict the digit
        #st.subheader("Predicted Number new model: new_mnist(4x4)_epoch70")
        st.subheader("Predicted Number : ")
        st.subheader(np.argmax(response))  # The predicted digit is the one with the highest probability       
        #st.header("Matrix contiaining response data for the new model")
        #st.write(response)
        if "correct_response" not in st.session_state:
            st.session_state.correct_response = np.argmax(response)

        if "is_wrong" not in st.session_state:
            st.session_state.is_wrong = False

        if st.button("Is the response wrong ? ", key='wrong_response'):
            st.session_state.is_wrong = True

        if st.session_state.is_wrong:
            st.session_state.correct_response = st.selectbox("Select the correct response", options=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            st.write(f"Correct response should have been {st.session_state.correct_response}")
        

        if st.button("Upload the image to Azure", key='upload'):
            upload(img, cnt , st.session_state.correct_response)
            st.session_state.correct_response = np.argmax(response)
            st.session_state.is_wrong = False
            st.experimental_rerun()

        


if __name__ == "__main__":
    st.set_page_config(
        page_title="Number Recognitions using ANN", page_icon=":pencil2:",
        layout="wide",initial_sidebar_state="expanded", menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }    
    )
    st.sidebar.subheader("Contents : ")
    main()
