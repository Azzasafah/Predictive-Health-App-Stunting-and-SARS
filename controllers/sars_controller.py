import numpy as np
from models.sars_model import load_models_sars

def predict_sars(form_data):
    # Ambil input dari form
    features = [float(form_data[key]) for key in ['jk', 'usia', 'demam', 'batuk', 'pilek', 'nyeri_otot', 'pneumonia', 'diare', 'infeksi_paru', 'isolasi']]
    X = np.array(features).reshape(1, -1)

    # Load semua model
    models = load_models_sars()

    results = {}

    # Logistic Regression
    logit_pred = models['logit'].predict(X)[0]
    logit_prob = models['logit'].predict_proba(X)[0]
    results['Logistic Regression'] = format_result(logit_pred, logit_prob)

    # Random Forest
    rf_pred = models['rf'].predict(X)[0]
    rf_prob = models['rf'].predict_proba(X)[0]
    results['Random Forest'] = format_result(rf_pred, rf_prob)

    return results

def format_result(predicted, score):
    if predicted == 1:
        result = 'Suspected Covid-19'
        confidence = score[0] * 100
    else:
        result = 'normal'
        confidence = score[1] * 100
    return {
        'result': result,
        'confidence': f"{confidence:.2f}%"
    }