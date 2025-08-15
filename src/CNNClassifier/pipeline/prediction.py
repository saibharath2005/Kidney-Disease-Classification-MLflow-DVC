import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        model_path = os.path.join("model", "model.h5")
        if not os.path.exists(model_path):
            return {"class": None, "confidence": 0, "error": "Model not found."}

        model = load_model(model_path)

        # Preprocess image
        test_image = image.load_img(self.filename, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        test_image = test_image / 255.0

        # Predict
        predictions = model.predict(test_image)
        class_index = np.argmax(predictions, axis=1)[0]
        confidence = float(np.max(predictions) * 100)

        classes = {0: "Normal", 1: "Cyst", 2: "Stone", 3: "Tumor"}
        predicted_class = classes.get(class_index, "Unknown")

        return {"class": predicted_class, "confidence": round(confidence, 2)}
