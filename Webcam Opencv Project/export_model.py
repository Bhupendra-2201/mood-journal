import os
from tensorflow.keras.models import model_from_json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_JSON = os.path.join(BASE_DIR, 'emotion_model1.json')
MODEL_H5   = os.path.join(BASE_DIR, 'emotion_model1.h5')
EXPORT_PATH = os.path.join(BASE_DIR, 'saved_model', '1')

print("Loading model from JSON and H5...")
with open(MODEL_JSON, 'r') as f:
    loaded_model_json = f.read()

model = model_from_json(loaded_model_json)
model.load_weights(MODEL_H5)

print("Exporting to SavedModel format...")
model.save(EXPORT_PATH, save_format='tf')
print(f"Model exported successfully to {EXPORT_PATH}")
