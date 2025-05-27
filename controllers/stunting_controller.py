import numpy as np
from models.stunting_model import load_models_stunting

def predict_stunting(form_data):
    # Ambil input dari form
    features = [float(form_data[key]) for key in ['jk', 'bbl', 'tbl', 'usiau', 'bb', 'tb']]
    X = np.array(features).reshape(1, -1)

    # Load semua model
    models = load_models_stunting()

    results = {}

    # Logistic Regression
    logit_pred = models['logit'].predict(X)[0]
    logit_prob = models['logit'].predict_proba(X)[0]
    results['Logistic Regression'] = format_result(logit_pred, logit_prob)

    # Random Forest
    rf_pred = models['rf'].predict(X)[0]
    rf_prob = models['rf'].predict_proba(X)[0]
    results['Random Forest'] = format_result(rf_pred, rf_prob)

    # K-Nearest Neighbors
    knn_pred = models['knn'].predict(X)[0]
    knn_prob = models['knn'].predict_proba(X)[0]
    results['K-Nearest Neighbors'] = format_result(knn_pred, knn_prob)
    
    # Gaussian Naive Bayes
    gnb_pred = models['gnb'].predict(X)[0]
    gnb_prob = models['gnb'].predict_proba(X)[0]
    results['Gaussian Naive Bayes'] = format_result(gnb_pred, gnb_prob)

    return results

def format_result(predicted, score):
    if predicted == 0:
        result = 'Normal'
        confidence = score[0] * 100
    elif predicted == 1:
        result = 'Stunting'
        confidence = score[1] * 100
    else:
        result = 'Overweight'
        confidence = score[2] * 100
    return {
        'result': result,
        'confidence': f"{confidence:.2f}%"
    }