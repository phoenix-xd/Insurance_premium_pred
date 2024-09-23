from flask import Flask, request, render_template
import numpy as np
from src.pipelines.prediction_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/home')
def document():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/features')
def features():
    return render_template('features.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('predict.html')
    else:
        # Retrieve form data
        data = CustomData(
            age=int(request.form.get('age')),
            bmi=float(request.form.get('bmi')),
            children=int(request.form.get('children')),
            sex=request.form.get('sex'),
            smoker=request.form.get('smoker'),
            region=request.form.get('region')  # Match with the form name
        )
        
        # Convert form data to DataFrame
        final_new_data = data.get_data_as_dataframe()
        
        # Predict using the pipeline
        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(final_new_data)
        
        # If `pred` is an array or list, round the first element
        results = round(pred[0]) if isinstance(pred, (list, np.ndarray)) else round(pred)
        
        return render_template('prediction.html', final_result=results , age=data.age , bmi=data.bmi ,children=data.children, sex=data.sex, smoker= data.smoker, region= data.region)

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
