import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os


class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename
        # Define the class labels in the same order as your model outputs
        self.class_labels = ["Class_A", "Class_B", "Class_C", "Class_D"]  
        # Example: replace these names with your actual 4 class names
        # e.g. ["Normal", "Tumor_Type1", "Tumor_Type2", "Other"]

    def predict(self):
        # Load the trained model
        model = load_model(os.path.join("model", "model.h5"))

        # Load and preprocess the image
        test_image = image.load_img(self.filename, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)  # Shape: (1, 224, 224, 3)

        # Get predictions
        predictions = model.predict(test_image)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_label = self.class_labels[predicted_class_index]

        # Print and return result
        print(f"Predicted Class Index: {predicted_class_index}")
        print(f"Predicted Label: {predicted_label}")

        return [{"image": predicted_label}]
