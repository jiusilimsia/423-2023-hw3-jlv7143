import os
import yaml
from pathlib import Path
import logging
import logging.config
import streamlit as st
import pickle
import numpy as np
import pandas as pd
from PIL import Image

import src.generate_features as gf
import src.aws_utils as aws



# Configuration, data, directory ====================================
# Load logging configuration file
logging.config.fileConfig("configs/logging/local.conf")
logger = logging.getLogger("clouds")

# Load configuration file
try:
    with open('configs/default-config.yaml', 'r') as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    logger.error("Configuration file not found.")
    raise
except Exception as e:
    logger.error("Error loading configuration file: %s", e)
    raise

# Load the Cloud dataset and trained classifier
BUCKET_NAME = os.getenv("BUCKET_NAME", "jlv7143-clouds-app")
ARTIFACTS_PREFIX = Path(os.getenv("ARTIFACTS_PREFIX", "artifacts/"))

# Create artifacts directory to keep model files
artifacts = Path() / "artifacts"
artifacts.mkdir(exist_ok=True)



# Define functions ==================================================
@st.cache_data(show_spinner=False)
def load_image(image_path):
    """
    Opens an image file and returns an Image object.
    Parameters:
        image_path (str): The path to the image file.
    Returns:
        PIL.Image.Image: An image object
    """
    img = Image.open(image_path)
    return img


@st.cache_data
def load_model_versions(path):
    """
    This function gets the names of all available model versions.
    Parameters:
        path (Path): Path to the directory containing model versions.
    Returns:
        list of str: Names of available model versions.
    """
    return [p.name for p in path.glob("*") if p.is_dir()]


@st.cache_data
def load_data(data_file, s3_key):
    """
    This function loads the data for the app.
    Parameters:
        data_file (Path): Path to the local data file.
        s3_key (str): Key to the data file in the S3 bucket.
    Returns:
        tuple: Tuple containing the class names, feature matrix, and column names.
    """
    logger.info(f"Loading artifacts from: {artifacts.absolute()}") 
    # Download files from S3 to local
    aws.download_s3(BUCKET_NAME, s3_key, cloud_file)
    # Load files into memory
    try:
        cloud = pd.read_csv(data_file)
    except FileNotFoundError:
        logger.error("Data file not found: %s", data_file)
        raise
    except Exception as e:
        logger.error("Error reading data file: %s", e)
        raise
    X: np.ndarray = cloud.drop('class', axis=1).values  # assuming 'class' column holds the target values
    class_names = cloud['class'].unique()
    column_names = list(cloud.columns)

    return class_names, X, column_names


@st.cache_resource
def load_model(model_file, model_s3_key):
    """
    This function loads the model for the app.
    Parameters:
        model_file (Path): Path to the local model file.
        model_s3_key (str): Key to the model file in the S3 bucket.
    Returns:
        The loaded model.
    """
    # Download files from S3 to local
    aws.download_s3(BUCKET_NAME, model_s3_key, model_file)
    # Load model into memory
    try:
        with open(model_file, 'rb') as f:
            clf = pickle.load(f)
    except FileNotFoundError:
        logger.error(f"Model file not found: {model_file}")
        raise
    except pickle.UnpicklingError as e:
        logger.error(f"Error loading the model: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise
    return clf


def slider_values(series) -> tuple[float, float, float]:
    """
    This function computes the values for the slider UI elements.
    Parameters:
        series (Series): Series containing the feature values.
    Returns:
        tuple: Tuple containing the min, max, and mean of the feature values.
    """
    if series.size == 0:
        raise ValueError("Input series is empty.")
    return (
        float(series.min()),
        float(series.max()),
        float(series.mean()),
    )



# App ===============================================================
# Create the application title and description
st.title("Cloud Class Prediction")
st.write("This app predicts the class of a cloud based on its properties.")

st.subheader("Model Selection")
# Default model version: logistic
model_version = os.getenv("DEFAULT_MODEL_VERSION", "logistic-model")
# Find available model versions in artifacts dir
available_models = load_model_versions(artifacts)
# Create a dropdown to select the model
model_version = st.selectbox("Select Model", list(available_models))
st.write(f"Selected model version: {model_version}")


# Establish the dataset and TMO locations based on selection
cloud_file = artifacts / model_version / "cloud_data.csv"
cloud_model_file = artifacts / model_version / "cloud_classifier.pkl"

# Configure S3 location for each artifact
cloud_s3_key = str(ARTIFACTS_PREFIX / model_version / cloud_file.name)
cloud_model_s3_key = str(ARTIFACTS_PREFIX / model_version / cloud_model_file.name)

# Load the dataset and TMO into memory
class_names, X, column_names = load_data(cloud_file, cloud_s3_key)
clf = load_model(cloud_model_file, cloud_model_s3_key)


# Find the index of Variables
IR_min_index = column_names.index("IR_min")
IR_max_index = column_names.index("IR_max")
IR_mean_index = column_names.index("IR_mean")
visible_entropy_index = column_names.index("visible_entropy")
visible_contrast_index = column_names.index("visible_contrast")

# Sidebar inputs for feature values
st.sidebar.header("Input Parameters")

st.sidebar.markdown('**IR Min**')
IR_min = st.sidebar.slider("Minimum value of Infrared Radiation",*slider_values(X[:, IR_min_index]))
st.sidebar.markdown('**IR Max**')
IR_max = st.sidebar.slider("Maximum value of Infrared Radiation", *slider_values(X[:, IR_max_index]))
st.sidebar.markdown('**IR Mean**')
IR_mean = st.sidebar.slider("Mean value of Infrared Radiation", *slider_values(X[:, IR_mean_index]))
st.sidebar.markdown('**Visible entropy**')
visible_entropy = st.sidebar.slider("Randomness/disorder in the visible light spectrum", *slider_values(X[:, visible_entropy_index]))
st.sidebar.markdown('**Visible contrast**')
visible_contrast = st.sidebar.slider("Difference between the brightest and darkest parts", *slider_values(X[:, visible_contrast_index]))

# Save the input data into dataframe format
input_data = [[IR_min, IR_max, IR_mean, visible_entropy, visible_contrast]]
input_data_df = pd.DataFrame(input_data, columns=['IR_min', 'IR_max', 'IR_mean', 'visible_entropy', 'visible_contrast'])
# Transform the input data
features = gf.generate_features(input_data_df, config["generate_features"])
# Adjust the order of the columns
features = features[config["features"]]


# Make predictions on user inputs
prediction = clf.predict(features)
predicted_class = class_names[int(prediction[0])]

# Display the predicted class and probability
st.subheader("Prediction")
st.write(f"Predicted Cloud Class: {predicted_class}")
st.write(f"Probability: {clf.predict_proba(features)[0][int(prediction[0])]:.2f}")

# Load cloud image
image_path = "sunshine-and-blue-skies-ahead.webp"
img = load_image(image_path)
st.image(img)
