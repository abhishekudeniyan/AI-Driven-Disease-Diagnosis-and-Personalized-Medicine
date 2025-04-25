from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')
from chatbot import getAnswer  # Ensure this import is correct

# Initialize Flask app
app = Flask(__name__)

# Load the trained Logistic Regression model
model = joblib.load('Multinomial Naive Baye_model.pkl')

# Load the dataset for symptoms and diseases
data1 = pd.read_csv('Final_Updated__Dataset.csv')

# Prepare data (Symptom and Disease dictionaries)
X = data1.drop("diseases", axis=1)
y = data1['diseases']

symptoms_dict = {col: idx + 1 for idx, col in enumerate(data1.columns[:-1])}

def given_predicted_value(user_symptoms):
    # Initialize a zero vector for the input based on the number of symptoms
    input_vector = np.zeros(len(symptoms_dict))
    
    unrecognized_symptoms = []  # List to hold unrecognized symptoms

    # Iterate over user input symptoms and fill the input vector
    for item in user_symptoms:
        if item in symptoms_dict:
            input_vector[symptoms_dict[item] - 1] = 1  # Mark the symptom as present
        else:
            unrecognized_symptoms.append(item)  # Collect unrecognized symptoms
    
    if unrecognized_symptoms:
        return f"Unrecognized symptoms: {', '.join(unrecognized_symptoms)}"
        
    # Make a prediction using the model
    try:
        prediction = model.predict([input_vector])[0]
        print(f"Predicted Disease: {prediction}")
        return prediction
    except Exception as e:
        print(f"Prediction error: {e}")
        return "Unknown disease"  # Return in case of a prediction error

# Home route
@app.route('/')
def home():
    # Pass the symptoms_dict to the template
    symptoms = list(symptoms_dict.keys())
    return render_template('index.html', symptoms=symptoms)
    
# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the list of symptoms from the form
        symptoms = request.form.getlist('symptoms')  
        user_symptoms = [s.strip().lower() for s in symptoms]  # Clean up symptom inputs

        # Get the predicted disease or unrecognized symptoms
        predicted_result = given_predicted_value(user_symptoms)  

        if "Unrecognized symptoms" in str(predicted_result) or "Unknown disease" in str(predicted_result):
            prediction = predicted_result  # Display the error message (unrecognized symptoms/unknown disease)
            answer = "No medication information available."  # No medication info for unrecognized cases
        else:
            # If a valid disease is predicted
            disease = predicted_result  # Get the disease name
            prediction = f"Disease: {disease}".capitalize()  # Format the disease prediction
            answer = getAnswer(disease)  # Fetch the corresponding medication/answer

        # Re-populate the form with symptoms
        symptoms_list = list(symptoms_dict.keys())

        # Render the template with the prediction and medication answer
        return render_template('index.html', 
                               prediction=prediction, 
                               answer=answer,  # Pass the medication info
                               symptoms=symptoms_list, 
                               selected_symptoms=symptoms)

if __name__ == '__main__':
    app.run(debug=True)
