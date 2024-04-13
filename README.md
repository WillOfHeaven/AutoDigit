## Handwritten Digit Recognizer with TensorFlow & Keras

This repository contains the code for a project that explores the use of Convolutional Neural Networks (CNNs) for handwritten digit recognition using TensorFlow and Keras. The project focuses on building a proof-of-concept system with a high degree of accuracy.

### Project Goals

* Develop a CNN model to classify handwritten digits with high accuracy.
* Utilize TensorFlow and Keras for model training and evaluation.
* Employ readily available tools for local development. (This project uses MLflow locally)

### Prerequisites

* Python 3.x
* TensorFlow
* Keras
* NumPy
* Matplotlib (optional, for visualization)
* MLflow (optional, for experiment tracking)

### Installation

1. Clone this repository:

```bash
git clone https://github.com/[your-username]/handwritten-digit-recognizer.git
```

2. Navigate to the project directory:

```bash
cd handwritten-digit-recognizer
```

3. Install the required dependencies (assuming you have pip):

```bash
pip install -r requirements.txt
```


### Usage

1. **Data Preparation:**

    * The project currently does not include a dataset. You will need to obtain a suitable handwritten digit dataset (e.g., MNIST) and place it in the `data` folder.
    * Ensure the data is preprocessed and formatted appropriately for the model (e.g., normalization, reshaping).

2. **Model Training:**

    * Run the `train.py` script to train the model. This script will:
        * Load the dataset.
        * Preprocess the data (if not already done).
        * Build and train the CNN model.
        * Evaluate the model performance.
        * (Optional) Utilize MLflow for experiment tracking (requires MLflow installation).

3. **Evaluation:**

    * The `train.py` script will print out the model's evaluation metrics (e.g., accuracy, precision, recall).
    * You can also use additional scripts or tools (e.g., Matplotlib) to visualize the model's performance.

4. **Future Considerations (Not Included in this Version):**

    * Implementing a mechanism for automated retraining upon discovery of new training data.
    * Exploring cloud-based deployment options on platforms like Azure Machine Learning.


### Contributing

We welcome contributions to this project! Please feel free to submit pull requests with improvements or additional features.

### License

This project is licensed under the MIT License. See the LICENSE file for details.

### Disclaimer

This is a student project and is not intended for production use.
