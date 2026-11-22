import os
from pathlib import Path

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import cv2
import numpy as np
import tensorflow as tf

SCRIPT_DIR = Path(__file__).resolve().parent


def load_labels(path):
    labels = []
    with open(path, "r", encoding="utf-8") as label_file:
        for line_number, line in enumerate(label_file, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(f"Invalid label on line {line_number}: {line!r}")
            labels.append(parts[1])
    return labels


def load_model():
    model_path = SCRIPT_DIR / "keras_Model.h5"
    return tf.keras.models.load_model(str(model_path), compile=False)


def capture_one_frame(camera_index=0):
    camera = cv2.VideoCapture(camera_index)
    try:
        if not camera.isOpened():
            raise RuntimeError("Could not open the webcam.")
        captured, frame = camera.read()
    finally:
        camera.release()

    if not captured or frame is None:
        raise RuntimeError("Could not capture a frame from the webcam.")
    return frame


def preprocess_frame(model, frame):
    _, height, width, _ = model.input_shape
    frame = cv2.cvtColor(cv2.resize(frame, (width, height)), cv2.COLOR_BGR2RGB)
    image = np.asarray(frame, dtype=np.float32)
    image = (image / 127.5) - 1
    return np.expand_dims(image, axis=0), frame


def predict_frame(model, frame, labels):
    image, ai_input = preprocess_frame(model, frame)
    predictions = model.predict(image, verbose=0)[0]
    predicted_index = int(np.argmax(predictions))
    if predicted_index >= len(labels):
        raise ValueError("The model returned a class with no corresponding label.")

    return labels[predicted_index], float(predictions[predicted_index]), ai_input


def main():
    model = load_model()
    labels = load_labels(SCRIPT_DIR / "labels.txt")
    frame = capture_one_frame()
    predicted_class, confidence, _ = predict_frame(model, frame, labels)

    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.2%}")


if __name__ == "__main__":
    main()
