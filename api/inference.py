import numpy as np
import tensorflow as tf
from PIL import Image
from io import BytesIO
import cv2
import base64

MODEL_PATH = "model/covid_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)
CLASS_NAMES = ["Covid", "Normal", "Viral Pneumonia"]

def preprocess_image(image_bytes):
    """Preprocess uploaded image to model input."""
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def get_last_conv_layer(model):
    """Find the last Conv2D layer in the model."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in the model")

def gradcam(image_bytes):
    img = preprocess_image(image_bytes)
    last_conv_name = get_last_conv_layer(model)
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img)
        loss = predictions[:, tf.argmax(predictions[0])]

    grads = tape.gradient(loss, conv_outputs)

    # Reduce gradients over existing dimensions
    if len(conv_outputs.shape) == 4:
        pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
        conv_outputs = conv_outputs[0]
    elif len(conv_outputs.shape) == 2:
        pooled_grads = tf.reduce_mean(grads, axis=0)
        conv_outputs = conv_outputs[0]
    else:
        raise ValueError(f"Unexpected conv layer shape: {conv_outputs.shape}")

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Superimpose on original image
    original_img = cv2.cvtColor(np.array(Image.open(BytesIO(image_bytes)).resize((224,224))), cv2.COLOR_RGB2BGR)
    superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)

    _, buffer = cv2.imencode('.png', superimposed_img)
    return base64.b64encode(buffer).decode('utf-8')

def predict_image(file_bytes):
    img_array = preprocess_image(file_bytes)
    preds = model.predict(img_array)
    class_index = np.argmax(preds[0])
    confidence = round(float(preds[0][class_index]) * 100, 2)
    
    all_probs = {CLASS_NAMES[i]: round(float(preds[0][i])*100,2) for i in range(len(CLASS_NAMES))}
    gradcam_img = gradcam(file_bytes)

    return {
        "prediction": CLASS_NAMES[class_index],
        "confidence": confidence,
        "all_probs": all_probs,
        "gradcam": gradcam_img
    }
