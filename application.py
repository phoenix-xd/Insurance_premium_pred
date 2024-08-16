from flask import Flask, request, render_template
from src.pipelines.prediction_pipeline import CustomData, PredictPipeline
import numpy as np

application = Flask(__name__)
app = application

@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('form.html')
    else:
        # Retrieve form data
        age = request.form.get('age')
        bmi = request.form.get('bmi')
        children = request.form.get('children')
        sex = request.form.get('sex')
        smoker = request.form.get('smoker')
        region = request.form.get('region_categories')

        # Check if all form fields are filled
        if not all([age, bmi, children, sex, smoker, region]):
            return render_template('form.html', error="Please fill all fields")

        try:
            age = int(age)
            bmi = float(bmi)
            children = int(children)

            # Create CustomData object
            data = CustomData(
                age=age,
                bmi=bmi,
                children=children,
                sex=sex,
                smoker=smoker,
                region=region
            )

            # Convert form data to DataFrame
            final_new_data = data.get_data_as_dataframe()

            # Predict using the pipeline
            predict_pipeline = PredictPipeline()
            pred = predict_pipeline.predict(final_new_data)

            # If `pred` is an array or list, round the first element
            if isinstance(pred, (list, np.ndarray)):
                results = round(pred[0])
            else:
                results = round(pred)

            return render_template('form.html', final_result=results)

        except ValueError:
            return render_template('form.html', error="Invalid input. Please enter valid numbers for age, bmi, and children.")
if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)