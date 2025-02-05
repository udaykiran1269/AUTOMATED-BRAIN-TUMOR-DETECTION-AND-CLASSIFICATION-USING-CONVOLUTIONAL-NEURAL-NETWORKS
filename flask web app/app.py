import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)


# Load the pre-trained model
model = tf.keras.models.load_model('best_weights.h5')  # Replace with the actual path to your model

# Define the path to the uploaded images directory
UPLOAD_FOLDER = 'static/uploaded_images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
past_results = []
predicted_class=''

@app.route('/', methods=['GET', 'POST'])

def index():
    global predicted_class 
    global past_results
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return render_template('index.html', message='No file part')

        file = request.files['file']

        # If the user submits an empty form
        if file.filename == '':
            return render_template('index.html', message='No selected file')

        if file:
            # Save the uploaded file to the UPLOAD_FOLDER
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Load and preprocess the image
            img = load_img(file_path, target_size=(150, 150), color_mode="grayscale")
            img = img_to_array(img)
            image_tensor = np.expand_dims(img, axis=0)
            image_tensor /= 255.0  # Normalize the image

            # Make a prediction with the loaded model
            predictions = model.predict(image_tensor)
            past_results.insert(0, predicted_class)
            # Keep only the most recent 10 results
            past_results = past_results[:10]

            # Map the prediction to a class label
            class_labels = ['Normal', 'Glioma', 'Meningioma', 'Pituitary']
            predicted_class = class_labels[np.argmax(predictions)]
            image_url = f"/{app.config['UPLOAD_FOLDER']}/{file.filename}"
        return render_template('index.html', message='Image uploaded and classified as:', prediction=predicted_class, image_url=image_url, result_class='result', past_results=past_results)

    return render_template('index.html', message='Upload an image for classification.', result_class='', past_results=[])


if __name__ == '__main__':
    app.run(debug=True)

