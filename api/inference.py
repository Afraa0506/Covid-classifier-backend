import numpy as np
import tensorflow as tf
from PIL import Image
from io import BytesIO
import cv2
import base64
import os
import gdown

MODEL_PATH = "model/covid_model.h5"
MODEL_DIR = "model"

# Use confirm=t to bypass large file restrictions
FILE_ID = "1HbSuKd0Lij3ptqkRNAxQQfT_ONKodEth"
MODEL_URL = f"https://drive.google.com/uc?id={FILE_ID}&confirm=t"

# Ensure model folder exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Download model if not present
if not os.path.exists(MODEL_PATH):
    print("Model not found! Downloading from Google Drive...")
    try:
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        print("Model download complete!")
    except Exception as e:
        print(f"ERROR DOWNLOADING MODEL: {e}")
        raise RuntimeError("Model download failed.")

# Load model safely
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
model.build(input_shape=(None, 224, 224, 3))

# Warm-up prediction to avoid "model never called" error
_ = model.predict(np.zeros((1, 224, 224, 3)))
print("Model loaded successfully!")

CLASS_NAMES = ["Covid", "Normal", "Viral Pneumonia"]


# Preprocessing function
def preprocess_image(image_bytes):
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# Grad-CAM utilities
def get_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found for Grad-CAM.")


def gradcam(image_bytes):
    img = preprocess_image(image_bytes)
    last_conv = get_last_conv_layer(model)

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)

    # Handle shapes for different model architectures
    if len(conv_outputs.shape) == 4:
        conv_outputs = conv_outputs[0]
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    else:
        conv_outputs = conv_outputs[0]
        pooled_grads = tf.reduce_mean(grads, axis=0)

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()

    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    original = cv2.cvtColor(
        np.array(Image.open(BytesIO(image_bytes)).resize((224, 224))),
        cv2.COLOR_RGB2BGR
    )

    superimposed = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)
    _, buffer = cv2.imencode(".png", superimposed)

    return base64.b64encode(buffer).decode("utf-8")


# Prediction API
def predict_image(file_bytes):
    img_array = preprocess_image(file_bytes)
    preds = model.predict(img_array)

    class_index = int(np.argmax(preds[0]))
    confidence = round(float(preds[0][class_index]) * 100, 2)

    all_probs = {
        CLASS_NAMES[i]: round(float(preds[0][i]) * 100, 2)
        for i in range(len(CLASS_NAMES))
    }

    heatmap = gradcam(file_bytes)

    return {
        "prediction": CLASS_NAMES[class_index],
        "confidence": confidence,
        "all_probs": all_probs,
        "gradcam": heatmap
    }
