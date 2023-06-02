import logging
import pandas as pd
import numpy as np


# Set up the logger
logger = logging.getLogger(__name__)


def log_transform(data: pd.DataFrame, log_transform_config: dict) -> pd.DataFrame:
    """
    Perform log transformation for specified columns in the configuration.

    Args:
        data (pd.DataFrame): The input dataset.
        log_transform_config (dict): A dictionary containing the log transformation configuration.

    Returns:
        pd.DataFrame: The processed dataset with log-transformed features.
    """
    try:
        for new_feature, col in log_transform_config.items():
            data[new_feature] = data[col].apply(np.log)
        logger.debug("Feature log transformation(s) finished")
    except KeyError as e:
        logger.error("Wrong column name for the data: %s", e)
        raise
    except Exception as e:
        logger.error("Error while performing log transformation: %s", e)
        raise

    return data


def multiply(data: pd.DataFrame, multiply_config: dict) -> pd.DataFrame:
    """
    Perform multiplication for specified column pairs in the configuration.

    Args:
        data (pd.DataFrame): The input dataset.
        multiply_config (dict): A dictionary containing the multiplication configuration.

    Returns:
        pd.DataFrame: The processed dataset with multiplied features.
    """
    try:
        for new_feature, cols in multiply_config.items():
            data[new_feature] = data[cols["col_a"]].multiply(data[cols["col_b"]])
        logger.debug("Feature multiplication(s) finished")
    except KeyError as e:
        logger.error("Wrong column name for the data: %s", e)
        raise
    except Exception as e:
        logger.error("Error while performing multiplication: %s", e)
        raise

    return data


def calculate_range(data: pd.DataFrame, range_config: dict) -> pd.DataFrame:
    """
    Calculate the range for specified columns in the configuration.

    Args:
        data (pd.DataFrame): The input dataset.
        range_config (dict): A dictionary containing the range calculation configuration.

    Returns:
        pd.DataFrame: The processed dataset with range features.
    """
    try:
        for feature, cols in range_config.items():
            data[feature] = data[cols["max_col"]] - data[cols["min_col"]]
        logger.debug("Feature range calculation(s) finished")
    except KeyError as e:
        logger.error("Wrong column name for the data: %s", e)
        raise
    except Exception as e:
        logger.error("Error while calculating range: %s", e)
        raise

    return data


def calculate_norm_range(data: pd.DataFrame, norm_range_config: dict) -> pd.DataFrame:
    """
    Calculate the normalized range for specified columns in the configuration.

    Args:
        data (pd.DataFrame): The input dataset.
        norm_range_config (dict): A dictionary containing the normalized range calculation configuration.

    Returns:
        pd.DataFrame: The processed dataset with normalized range features.
    """
    try:
        for new_feature, cols in norm_range_config.items():
            data[new_feature] = (data[cols["max_col"]] - data[cols["min_col"]]).divide(data[cols["mean_col"]])
        logger.debug("Feature normalized range calculation(s) finished")
    except KeyError as e:
        logger.error("Wrong column name for the data: %s", e)
        raise
    except Exception as e:
        logger.error("Error while calculating normalized range: %s", e)
        raise

    return data


def generate_features(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Generates features for a dataset using a configuration dictionary.

    Args:
        data (pd.DataFrame): The input dataset.
        config (dict): A dictionary containing the feature generation configuration.

    Returns:
        pd.DataFrame: The processed dataset with the new features and the target variable.
    """
    try:
        data = log_transform(data, config["log_transform"])
        data = multiply(data, config["multiply"])
        data = calculate_range(data, config["calculate_range"])
        data = calculate_norm_range(data, config["calculate_norm_range"])

        logger.info("Features generated successfully.")
    except Exception as e:
        logger.error("Error while generating features: %s", e)
        raise

    return data


def save_dataset(data: pd.DataFrame, save_data_path: str):
    """
    Save the dataset to a CSV file.

    Args:
        data: A pandas DataFrame containing the dataset.
        save_data_path: The path where the CSV file should be saved.
    """

    try:
        data.to_csv(save_data_path, index=False)
        logger.info("Dataset saved successfully at %s", save_data_path)
    except FileNotFoundError as e:
        logger.error("Error while saving the dataset: %s", e)
        logger.error("File path not found: %s", save_data_path)
        raise
    except PermissionError as e:
        logger.error("Error while saving the dataset: %s", e)
        logger.error("Permission denied: %s", e)
        raise
    except Exception as e:
        logger.error("Error while saving the dataset: %s", e)
        raise
