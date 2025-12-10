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
MODEL_URL = "https://drive.google.com/uc?export=download&id=1HbSuKd0Lij3ptqkRNAxQQfT_ONKodEth"

os.makedirs(MODEL_DIR, exist_ok=True)

# Download model if missing
if not os.path.exists(MODEL_PATH):
    print("Model not found! Downloading model from Google Drive...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    print("Model download complete!")

# Load the model
model = tf.keras.models.load_model(MODEL_PATH)

# IMPORTANT FIX — warmup call to define model.input
_ = model.predict(np.zeros((1, 224, 224, 3)))

CLASS_NAMES = ["Covid", "Normal", "Viral Pneumonia"]


def preprocess_image(image_bytes):
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)


def get_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in model")


def gradcam(image_bytes):
    img = preprocess_image(image_bytes)
    last_conv = get_last_conv_layer(model)

    grad_model = tf.keras.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= (heatmap.max() + 1e-8)

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


def predict_image(file_bytes):
    img_array = preprocess_image(file_bytes)
    preds = model.predict(img_array)

    class_index = np.argmax(preds[0])
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
