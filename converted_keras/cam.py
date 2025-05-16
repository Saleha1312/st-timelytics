import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("keras_Model.h5")  # Replace with your model path
    return model

model = load_model()

# Function to preprocess the image
def preprocess_image(image: Image.Image):
    image = image.resize((224, 224))  # Adjust size based on your model
    image = np.array(image) / 255.0   # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit UI
st.title("Anomaly Detection - Teachable Machine Model")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    processed_img = preprocess_image(img)
    prediction = model.predict(processed_img)[0]

    predicted_class = np.argmax(prediction)
    confidence = prediction[predicted_class]

    class_names = ["Good", "Anomaly"]  # Change if needed

    st.write(f"*Prediction*: {class_names[predicted_class]}")
    st.write(f"*Confidence*: {confidence:.2f}")


# import streamlit as st
# # import tensorflow as tf
# from keras.models import load_model  # TensorFlow is required for Keras to work
# import cv2  # Install opencv-python
# import numpy as np

# # Disable scientific notation for clarity
# np.set_printoptions(suppress=True)

# # Load the model
# model = load_model("keras_Model.h5", compile=False)

# # Load the labels
# class_names = open("labels.txt", "r").readlines()

# # CAMERA can be 0 or 1 based on default camera of your computer
# camera = cv2.VideoCapture(0)

# while True:
#     # Grab the webcamera's image.
#     ret, image = camera.read()

#     # Resize the raw image into (224-height,224-width) pixels
#     image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

#     # Show the image in a window
#     cv2.imshow("Webcam Image", image)

#     # Make the image a numpy array and reshape it to the models input shape.
#     image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

#     # Normalize the image array
#     image = (image / 127.5) - 1

#     # Predicts the model
#     prediction = model.predict(image)
#     index = np.argmax(prediction)
#     class_name = class_names[index]
#     confidence_score = prediction[0][index]

#     # Print prediction and confidence score
#     print("Class:", class_name[2:], end="")
#     print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

#     # Listen to the keyboard for presses.
#     keyboard_input = cv2.waitKey(1)

#     # 27 is the ASCII for the esc key on your keyboard.
#     if keyboard_input == 27:
#         break

# camera.release()
# cv2.destroyAllWindows()
