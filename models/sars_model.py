import joblib

def load_models_sars():
    return {
        "logit": joblib.load("model-development/lr_sars"),
        "rf"   : joblib.load("model-development/rf_sars"),
    }
