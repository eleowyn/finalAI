import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image

def load_model_and_classes():
    """
    Load trained model dan class indices
    """
    model_path = 'models/saved_models/padang_food_model.keras'
    class_indices_path = 'models/class_indices.json'
    
    # Cek apakah file model ada
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model tidak ditemukan di {model_path}. "
            "Jalankan training terlebih dahulu dengan: python cnn_train.py"
        )
    
    # Load model
    model = keras.models.load_model(model_path)
    
    # Load class indices
    if not os.path.exists(class_indices_path):
        raise FileNotFoundError(
            f"Class indices tidak ditemukan di {class_indices_path}. "
            "Pastikan training sudah selesai dengan benar."
        )
    
    with open(class_indices_path, 'r') as f:
        class_indices = json.load(f)
    
    # Reverse mapping: index -> class name
    idx_to_class = {v: k for k, v in class_indices.items()}
    
    return model, idx_to_class

def predict_image(img_path, model, idx_to_class, top_k=3):
    """
    Prediksi gambar dan return top-k predictions
    
    Args:
        img_path: Path ke gambar
        model: Loaded Keras model
        idx_to_class: Dictionary mapping index ke class name
        top_k: Jumlah top predictions yang akan dikembalikan
    
    Returns:
        List of dictionaries dengan format:
        [{'class': 'nama_makanan', 'confidence': 0.95}, ...]
    """
    IMG_SIZE = (224, 224)
    
    # Load dan preprocess gambar
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalisasi
    
    # Prediksi
    predictions = model.predict(img_array, verbose=0)[0]
    
    # Get top-k predictions
    top_indices = np.argsort(predictions)[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        results.append({
            'class': idx_to_class[idx],
            'confidence': float(predictions[idx])
        })
    
    return results