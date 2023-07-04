import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define the bug categories
categories = ['Functional', 'System', 'Usability', 'Security']

# Define descriptions and resources for each category
descriptions = {
    'Functional': 'A functional error refers to a malfunction or "bug" in the functions of a software application. This could be a feature that is not working as intended or a request that fails to process correctly. To resolve this, you might need to debug the application or check for updates that fix the issue.',
    'System': 'A system error refers to problems with the server or system where the application is running. This could be due to issues like low memory, full disk space, or server unavailability. Resolving this might involve freeing up memory or disk space, or checking the server status.',
    'Usability': 'A usability error refers to issues that prevent the user from interacting with the application effectively. This could be due to poor user interface design, confusing features, or lack of necessary functionality. To resolve this, you might need to redesign the user interface or add the necessary features.',
    'Security': 'A security error refers to vulnerabilities that could be exploited to gain unauthorized access to the system or data. This could be due to weak passwords, lack of encryption, or open ports. Resolving this might involve strengthening passwords, implementing encryption, or closing unnecessary ports.',
}

resources = {
    'Functional': 'https://www.softwaretestinghelp.com/functional-testing/',
    'System': 'https://www.softwaretestinghelp.com/system-testing/',
    'Usability': 'https://www.softwaretestinghelp.com/usability-testing/',
    'Security': 'https://www.softwaretestinghelp.com/security-testing/',
}

st.title('Bug Classifier')

# Allow the user to upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")

    IMG_SIZE = 224  # or whatever size you want


    # Preprocess the image
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = image.convert('RGB')  # Ensure the image is in RGB format
    image = np.array(image) / 255.0
    image = image.astype('float32')  # Convert the image to float32
    image = np.expand_dims(image, axis=0)

    # Set the tensor to point to the input data to be used for prediction
    interpreter.set_tensor(input_details[0]['index'], image)


    # Run the computations
    interpreter.invoke()

    # Get the output from the model
    predictions = interpreter.get_tensor(output_details[0]['index'])

    # Get the category with the highest prediction probability
    predicted_category = categories[np.argmax(predictions)]

    # Display the prediction
    st.write("The predicted category of the bug is: ", predicted_category)
    st.write("Description: ", descriptions[predicted_category])
    st.write("For more information, you can check this resource: ", resources[predicted_category])
