
# DISEASE PREDICTION MODEL TESTING - PYTHON FILE 



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load the saved logistics  model
best_nb_model= joblib.load('Multinomial Naive Baye_model.pkl')
print("Multinomial Naive Baye model loaded successfully.")



data1 = pd.read_csv('Final_Updated__Dataset.csv')

X= data1.drop("diseases", axis=1)
y= data1['diseases']

# Create a dictionary for symptoms 
symptoms_dict = {col: idx + 1 for idx, col in enumerate(data1.columns[:-1])}


def given_predicted_value(user_symptoms):
    # Initialize a zero vector for the input based on the number of symptoms
    input_vector = np.zeros(len(symptoms_dict))
    
    # Iterate over user input symptoms and fill the input vector
    for item in user_symptoms:
        if item in symptoms_dict:
            input_vector[symptoms_dict[item] - 1] = 1  # Mark the symptom as present
        else:
            print(f"Symptom '{item}' not recognized.")
            return None  # Exit if an unrecognized symptom is found
    
    # Make a prediction using the logistic regression model
    prediction = best_nb_model.predict([input_vector])[0]
   

    # Get the disease name from the reverse dictionary
    if prediction:
        print(f"Predicted Disease: {prediction}")
    else:
        print(f"Prediction '{prediction}' not found in the disease dictionary.")
        return "Unknown disease"



# Simulating user input
symptoms = input("Enter your symptoms (comma-separated): ").lower()

# Split the user's input into a list of symptoms (assuming they are comma-separated)
user_symptoms = [s.strip() for s in symptoms.split(',')]

print('Symptoms = ', user_symptoms)

# Get the predicted disease
given_predicted_value(user_symptoms)
