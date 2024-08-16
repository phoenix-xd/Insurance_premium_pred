import streamlit as st
import numpy as np
from src.pipelines.prediction_pipeline import CustomData, PredictPipeline

# Custom CSS for styling
st.markdown('''
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-kenU1KFdBIe4zVF0s0G1M5b4hcpxyD9F7jL+pmLZ4mk5R+6Zr9i5z4ET/tv0fgx" crossorigin="anonymous">
    <style>
        body {
            background-color: #f8f9fa;
            font-family: 'Roboto', sans-serif;
        }
        .container {
            max-width: 800px;
            margin-top: 50px;
            background: #ffffff;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0px 0px 20px rgba(0, 0, 0, 0.1);
        }
        .btn-primary {
            background-color: #007bff;
            border: none;
            font-size: 18px;
            font-weight: bold;
            padding: 10px 20px;
            border-radius: 10px;
            transition: background-color 0.3s ease;
            cursor: pointer;
        }
        .btn-primary:hover {
            background-color: #0056b3;
        }
        .form-label {
            font-weight: bold;
            color: #604cc3;
            margin-bottom: 5px;
        }
        .form-control {
            border-radius: 10px;
            border: 1px solid #ced4da;
            padding: 10px;
            font-size: 16px;
            color: #495057;
        }
        .result-box {
            background-color: #e9ecef;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
            font-size: 24px;
            font-weight: bold;
            text-align: center;
        }
        h1 {
            font-size: 36px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        p {
            font-size: 18px;
            color: #6c757d;
            margin-bottom: 20px;
        }
    </style>
''', unsafe_allow_html=True)

# App container
st.markdown('<div class="container">', unsafe_allow_html=True)

# App title and subtitle
st.markdown("<h1 class='text-center' style='color: #604cc3;'>Insurance Premium Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p class='text-center' style='color: #6c757d;'>Fill in the details below to predict your insurance premium.</p>", unsafe_allow_html=True)

# Form inputs with some alignment
with st.form("prediction_form"):
    st.markdown("<h4 style='color: #604cc3;'>Personal Information:</h4>", unsafe_allow_html=True)
    age = st.number_input('Please Enter Your Age', min_value=0, max_value=120, step=1, key="age", help="Enter your age in years", format="%d")
    bmi = st.number_input('Please Enter Your BMI', min_value=0.0, max_value=100.0, step=0.1, key="bmi", help="Enter your Body Mass Index (BMI)", format="%.1f")
    children = st.number_input('Number of Children', min_value=0, max_value=20, step=1, key="children", help="Enter the number of children you have", format="%d")

    st.markdown("<h4 style='color: #604cc3;'>Lifestyle Information:</h4>", unsafe_allow_html=True)
    sex = st.selectbox('Please Select Your Gender', ['male', 'female'], key="sex")
    smoker = st.selectbox('Are You a Smoker?', ['yes', 'no'], key="smoker")
    region = st.selectbox('Please Select Your Region', ['southeast', 'southwest', 'northeast', 'northwest'], key="region")
    
    submit_button = st.form_submit_button("Predict")

if submit_button:
    if not all([age, bmi, children, sex, smoker, region]):
        st.error("Please fill all fields")
    else:
        try:
            data = CustomData(
                age=int(age),
                bmi=float(bmi),
                children=int(children),
                sex=sex,
                smoker=smoker,
                region=region
            )

            final_new_data = data.get_data_as_dataframe()

            predict_pipeline = PredictPipeline()
            pred = predict_pipeline.predict(final_new_data)

            if isinstance(pred, (list, np.ndarray)):
                results = round(pred[0])
            else:
                results = round(pred)

            st.success(f"Predicted Insurance Premium: ${results}", icon="✅")

        except ValueError:
            st.error("Invalid input. Please enter valid numbers for age, BMI, and number of children.")

# Add an image
st.image("D:\projects PW\Insurance_premium_pred\inimg.jpg", caption="Insurance Concept")