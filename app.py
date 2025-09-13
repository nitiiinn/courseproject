import pickle 
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)
app=application

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        # Sanitize and validate inputs from form
        raw = {
            'gender': request.form.get('gender', '').strip(),
            'race_ethnicity': request.form.get('race_ethnicity', '').strip(),
            'parental_level_of_education': request.form.get('parental_level_of_education', '').strip(),
            'lunch': request.form.get('lunch', '').strip(),
            'test_preparation_course': request.form.get('test_preparation_course', '').strip(),
            'reading_score': request.form.get('reading_score', '').strip(),
            'writing_score': request.form.get('writing_score', '').strip()
        }

        # Required text fields
        required_text = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
        missing = [k for k in required_text if not raw.get(k)]
        # Validate numeric fields
        numeric_errors = []
        try:
            reading_score = float(raw['reading_score'])
        except Exception:
            numeric_errors.append('reading_score')
            reading_score = None
        try:
            writing_score = float(raw['writing_score'])
        except Exception:
            numeric_errors.append('writing_score')
            writing_score = None

        if missing or numeric_errors:
            err_parts = []
            if missing:
                err_parts.append(f"missing fields: {missing}")
            if numeric_errors:
                err_parts.append(f"invalid numeric fields: {numeric_errors}")
            error_message = '; '.join(err_parts)
            return render_template('home.html', error=error_message)

        data = CustomData(
            gender=raw['gender'],
            race_ethnicity=raw['race_ethnicity'],
            parental_level_of_education=raw['parental_level_of_education'],
            lunch=raw['lunch'],
            test_preparation_course=raw['test_preparation_course'],
            reading_score=reading_score,
            writing_score=writing_score
        )

        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html', results=results[0])
    
if __name__=="__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)