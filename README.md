# 🧠 AI-Driven Disease Diagnosis & Personalized Medication Recommendation System

This project is an AI-powered system designed to diagnose diseases based on user-input symptoms and provide **personalized medication** and **precaution recommendations** using machine learning models and LLMs like **Gemini Pro**.

---

## 📊 Project Summary

| Metric                     | Value                        |
|---------------------------|------------------------------|
| 🧬 Total Records           | 130,000+                     |
| 💡 Unique Symptoms         | 380                          |
| 🧫 Unique Diseases         | 770+                         |
| ⚖️ Dataset Type            | Imbalanced, Multi-Class      |
| 📈 Best Model Accuracy     | **93.28%** (Multinomial NB)  |
| ⏱️ Fastest Model           | Logistic Regression (90%)    |
| 🔍 Highest Precision       | Random Forest (86.3%)        |
| 🧠 LLM Integration         | Gemini Pro (for recommendations) |

---

## 🧠 Features

- 🔍 Disease prediction from symptoms (multi-class classification)
- 💊 Personalized medicine recommendations (Allopathy + Homeopathy)
- 📋 Precaution & diet suggestions per disease
- 📚 Knowledge-based outputs powered by Gemini Pro API
- 📈 Model comparison and selection
- 🛠️ Handles data imbalance using undersampling/oversampling

---

## 🚀 Models Used and Evaluation

### 1. **Multinomial Naive Bayes**
- ✅ Best performance on sparse binary features
- ✅ Final Accuracy: **93.28%**
- ⏱️ Fast training and prediction
- 🟢 Best for large imbalanced binary datasets

### 2. **Logistic Regression**
- ✔️ Accuracy: **89.74%**
- 🔄 Performs well with balanced sampling
- 🧪 Suitable for linear decision boundaries

### 3. **Random Forest**
- 🎯 Accuracy: **86.3%**
- 💡 High precision and recall
- 🐢 Slower, but better at handling nonlinear patterns

---

## 📁 Dataset Description

- Symptoms represented in binary (0 = absent, 1 = present)
- Many diseases have very few samples (imbalance)
- Applied techniques like:
  - **Undersampling** for majority classes
  - **SMOTE / Oversampling** for minority classes
  - Maximum 500 instances per class after balancing

---     https://www.kaggle.com/code/prathamjainzee/eda-disease-symptom-dataset-773

## 🛠️ Technologies Used

- Python 3.10
- Scikit-learn
- Pandas & NumPy
- Gemini Pro API (via Generative AI)
- Flask / Streamlit (Web Interface)
- Matplotlib / Seaborn (for analysis)

---
## 🧪 Sample Predictions

| Input Symptoms                          | Predicted Disease | Gemini Medication                   |
|-----------------------------------------|-------------------|-------------------------------------|
| fever, rash, joint pain                | Dengue            | Paracetamol, fluids                 |
| headache, stiff neck, fever            | Meningitis        | Ceftriaxone, rest                   |
| cough, chest pain, short breath        | Pneumonia         | Azithromycin, antibiotics           |

---

## 🔭 Future Scope

- Chatbot-based input collection
- Multi-modal support (image + symptoms)
- Voice-to-text interface using Whisper AI
- Auto-updated medicine database using public APIs

---

## 👤 Author

**Abhishek Udeniyan**  
M.Sc. Computer Science | AI Enthusiast | Data Science Researcher

---

## 📄 License

This project is licensed under the MIT License.  
Free to use and modify with proper credit.

## 🔧 Setup & Installation

```bash
# Clone the repo
git clone https://github.com/abhishekudeniyan/AI-Driven-Disease-Diagnosis-and-Personalized-Medicine.git

# Navigate into the project folder
cd AI-Driven-Disease-Diagnosis-and-Personalized-Medicine

# Install required libraries
pip install -r requirements.txt

# Run the application
streamlit run app.py
# or if using Flask
python app.py

---

