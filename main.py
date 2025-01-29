from flask import Flask, render_template, request, redirect, url_for, jsonify, session
import mysql.connector
import joblib
import tensorflow as tf
import numpy as np
from flask_cors import CORS
import os
import logging
from PIL import Image
import torchvision.transforms.functional as TF
import CNN
import torch
import pandas as pd

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Add a secret key for session management
CORS(app)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

db_config = {
    'host': 'localhost', 
    'user': 'root',  
    'password': 'Chandu2001*', 
    'database': 'agri'
}

# Load Models
crop_model = joblib.load('crop_recommendation_model.pkl')
fertilizer_model = joblib.load('fertilizer_recommendation_model.pkl')

DISEASE_CLASSES = {
    0: 'Healthy',
    1: 'Powdery Mildew',
    2: 'Leaf Blight',
}

disease_info = pd.read_csv('disease_info.csv', encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv', encoding='cp1252')

model = CNN.CNN(39)  
model.load_state_dict(torch.load("plant_disease_model_1_latest.pt"))
model.eval() 

def prediction(image_path):
    """ Predict the disease from the image. """
    try:
        image = Image.open(image_path)
        image = image.resize((224, 224))  # Resize image to the expected input size for the model
        input_data = TF.to_tensor(image)
        input_data = input_data.view((-1, 3, 224, 224))  # Add batch dimension
        output = model(input_data)
        output = output.detach().numpy()  
        index = np.argmax(output)  
        return index
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return None

# Context Processor
@app.context_processor
def inject_user():
    return dict(user=session.get('user'))

# Routes

@app.route('/')
def home():
    logger.info("Home page accessed")
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        phone = request.form['phone']
        try:
            # Connect to the database
            connection = mysql.connector.connect(**db_config)
            cursor = connection.cursor()

            # Check if the user is already registered
            cursor.execute("SELECT name FROM farmers WHERE email = %s", (email,))
            existing_user = cursor.fetchone()

            if existing_user:
                session['user'] = existing_user[0]
                logger.info(f"Existing farmer logged in: {existing_user[0]}")
            else:
                cursor.execute(
                    "INSERT INTO farmers (name, email, phone) VALUES (%s, %s, %s)",
                    (name, email, phone)
                )
                connection.commit()
                logger.info(f"New farmer registered: {name}")
                session['user'] = name  

        except mysql.connector.Error as err:
            logger.error(f"Error registering farmer: {err}")
            return f"Error: {err}"
        finally:
            cursor.close()
            connection.close()

        return redirect(url_for('home'))

    return render_template('registration.html')


@app.route('/recommend-crop', methods=['GET', 'POST'])
def crop_recommendation():
    if request.method == 'POST':
        data = [
            float(request.form['N']),
            float(request.form['P']),
            float(request.form['K']),
            float(request.form['temperature']),
            float(request.form['humidity']),
            float(request.form['ph']),
            float(request.form['rainfall'])
        ]
        prediction = crop_model.predict([data])
        logger.info(f"Crop recommendation data: {data} | Prediction: {prediction[0]}")
        return jsonify({'Recommended Crop': prediction[0]})
    return render_template('crop-recommendation.html')

@app.route('/recommend-fertilizer', methods=['GET', 'POST'])
def fertilizer_recommendation():
    if request.method == 'POST':
        try:
            temperature = float(request.form.get('temperature', 0))
            humidity = float(request.form.get('humidity', 0))
            moisture = float(request.form.get('moisture', 0))
            soil_type = int(request.form.get('soil_type', 1))
            crop_type = int(request.form.get('crop_type', 1))
            nitrogen = float(request.form.get('nitrogen', 0))
            potassium = float(request.form.get('potassium', 0))
            phosphorous = float(request.form.get('phosphorus', 0))  # Corrected field name

            logger.info(f"Received Form Data: Temperature: {temperature}, Humidity: {humidity}, Soil Moisture: {moisture}, Soil Type: {soil_type}, Crop Type: {crop_type}, Nitrogen (N): {nitrogen}, Potassium (K): {potassium}, Phosphorus (P): {phosphorous}")
            
            # Prepare the data for prediction
            data = [
                temperature,
                humidity,
                moisture,
                soil_type,
                crop_type,
                nitrogen,
                potassium,
                phosphorous
            ]
            
            # Get the prediction from the model
            prediction = fertilizer_model.predict([data])
            
            # Log the prediction
            logger.info(f"Fertilizer recommendation data: {data} | Prediction: {prediction[0]}")
            
            # Return the prediction as a JSON response
            return jsonify({'Recommended Fertilizer': prediction[0]})
        
        except Exception as e:
            # Handle any errors
            logger.error(f"Error during fertilizer recommendation: {str(e)}")
            return jsonify({'error': 'An error occurred during the prediction process.'}), 500

    # If the request is GET, render the form
    return render_template('fertilizer-recommendation.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    """ Handle image upload and prediction, return results as JSON. """
    if request.method == 'POST':
        # Get the uploaded image
        image = request.files['image']
        if not image:
            return jsonify({"error": "No file uploaded"}), 400
        
        # Save the image to a folder
        filename = image.filename
        file_path = os.path.join('static/uploads', filename)
        image.save(file_path)
        
        # Perform disease prediction
        pred = prediction(file_path)
        
        if pred is None:
            return jsonify({"error": "Prediction failed"}), 500
        
        pred=int(pred)
        # Retrieve disease information
        result = {
            "disease_name": disease_info['disease_name'][pred],
            "description": disease_info['description'][pred],
            "prevention_steps": disease_info['Possible Steps'][pred],
            "image_url": disease_info['image_url'][pred],
            "predicted_class_index": pred,
            "supplement_name": supplement_info['supplement name'][pred],
            "supplement_image_url": supplement_info['supplement image'][pred],
            "supplement_buy_link": supplement_info['buy link'][pred]
        }

        # Return the prediction data as a JSON response
        return jsonify(result)
    return render_template('disease-detection.html')

@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

# Test Routes for Debugging

@app.route('/test-crop', methods=['GET'])
def test_crop():
    return jsonify({"message": "Crop recommendation endpoint is working!"})

@app.route('/test-fertilizer', methods=['GET'])
def test_fertilizer():
    return jsonify({"message": "Fertilizer recommendation endpoint is working!"})

@app.route('/test-disease', methods=['GET'])
def test_disease():
    return jsonify({"message": "Disease detection endpoint is working!"})

if __name__ == '__main__':
    app.run(debug=True)
