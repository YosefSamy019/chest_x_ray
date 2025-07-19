import tensorflow as tf
import numpy as np
import cv2
import streamlit as st


CLASS_NAMES = ['NORMAL', 'PNEUMONIA']

def main():
    st.set_page_config(page_title="Chest X-ray", page_icon='🩻', layout="wide")
    st.title('Chest X-ray Detection')

    with st.expander('Readme'):
        with open("README.md", 'r', encoding='UTF-8') as f:
            st.write(f.read())

    cols = st.columns([0.6, 0.4])

    img_file = cols[0].file_uploader("Upload image", type=["jpg", 'png', 'jpeg'])


    if img_file is not None:
        cols[1].image(img_file, use_container_width=True)

        if cols[0].button('Predict', type='primary'):
            with st.spinner('Working...'):
                CLASS_NAMES = ['NORMAL', 'PNEUMONIA']
                COLORS = {
                    'NORMAL': '#00FF00',  # Green
                    'PNEUMONIA': '#FF0000'  # Red
                }

                # Preprocess
                img_clean = preprocess_image_from_file(img_file)

                # Predict
                prediction = predict_tflite(img_clean)

                # Get class index and label
                class_index = np.argmax(prediction, axis=1)[0]
                label = CLASS_NAMES[class_index]
                color = COLORS[label]

                # Show prediction nicely with color
                cols[0].markdown(
                    f"<h3 style='color:{color}'>Prediction: {label} {prediction[0,class_index]*100:0.2f}%</h3>",
                    unsafe_allow_html=True
                )

                # Optionally show raw probabilities
                cols[0].write(f"Probabilities ({CLASS_NAMES[0]}): {prediction[0,0]}")
                cols[0].write(f"Probabilities ({CLASS_NAMES[1]}): {prediction[0,1]}")


@st.cache_resource
def load_tflite_model(model_path=r"simple_cnn_v1_deploy.tflite"):
    """
    Load a TensorFlow Lite model from file.

    Args:
        model_path (str): Path to the .tflite model file.

    Returns:
        tf.lite.Interpreter: Loaded TFLite interpreter.
    """
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def predict_tflite(input_data):
    """
    Run inference with a TFLite interpreter.

    Args:
        interpreter (tf.lite.Interpreter): Loaded TFLite interpreter.
        input_data (np.ndarray): Input data for the model, properly shaped and dtype.

    Returns:
        np.ndarray: Output prediction from the model.
    """
    interpreter = load_tflite_model()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Ensure input matches model's expected dtype
    input_data = input_data.astype(input_details[0]['dtype'])

    # Add batch dimension if missing
    if len(input_data.shape) == 3:
        input_data = np.expand_dims(input_data, axis=0)

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])

    return output_data


def preprocess_image_from_file(img_file):
    IMG_W = 512
    IMG_H = 512

    # remove salt and pepper noise
    def apply_filter_median(image, kernel_size=5):
        return cv2.medianBlur(image, kernel_size).astype(image.dtype)

    def apply_sobel(image, dx=1, dy=0, ksize=3):
        return cv2.Sobel(image, cv2.CV_32F, dx, dy, ksize=ksize).astype(image.dtype)

    def apply_canny(image, threshold1=50, threshold2=60):
        return cv2.Canny(image, threshold1, threshold2).astype(image.dtype)

    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 0)

    # Resize the image to the target dimensions
    image = cv2.resize(image, (IMG_H, IMG_W))

    # Ensure the image has channel dimension
    if len(image.shape) == 2:
        image = np.reshape(image, (IMG_H, IMG_W, 1))

    image = apply_filter_median(image)

    image = np.stack((image, apply_sobel(image), apply_canny(image)), axis=-1)

    image = (image / 255.0).astype(np.float32)

    image = np.clip(image, 0.0, 1.0)

    return image.astype(np.float32)


if __name__ == '__main__':
    main()
