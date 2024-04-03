# extracting the data (data ingestion)
import tarfile
import os

def extract_tar_gz(file_path, extract_path):
    try:
        with tarfile.open(file_path, 'r:gz') as tar:
            tar.extractall(path=extract_path)
        print("Extraction successful.")

        os.remove('UrbanSound8K.tar.gz')
    except Exception as e:
        print("Extraction failed:", e)


from tensorflow.keras.models import load_model

def save_model(model, filename):
    """
    Save a Keras model to an HDF5 file.

    Parameters:
        model (tensorflow.keras.models.Model): The Keras model to save.
        filename (str): The filename (including path) to save the model to.
    """
    model.save(filename)
    print(f"Model saved to '{filename}'")

def load_saved_model(filename):
    """
    Load a Keras model from an HDF5 file.

    Parameters:
        filename (str): The filename (including path) to load the model from.

    Returns:
        tensorflow.keras.models.Model: The loaded Keras model.
    """
    model = load_model(filename)
    print(f"Model loaded from '{filename}'")
    return model

