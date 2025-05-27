import joblib

def load_models_stunting():
    return {
        "logit": joblib.load("model-development/lr_stunting"),
        "rf"   : joblib.load("model-development/rf_stunting"),
        "knn"  : joblib.load("model-development/knn_stunting"),
        "gnb"  : joblib.load("model-development/gnb_stunting"),
    }
