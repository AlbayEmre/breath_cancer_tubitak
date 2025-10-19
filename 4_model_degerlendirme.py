# Kaydedilmiş modelleri yükleyip test ve çapraz doğrulama ile en iyi modeli seçme
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_score
from joblib import load
# import tensorflow as tf
# from tensorflow import keras

# ==========================
# Veri seti
# ==========================
data_path = "data.csv"  # veri yolunu belirt
df = pd.read_csv(data_path)

drop_cols = [c for c in ['id', 'Unnamed: 32', 'Unnamed:32'] if c in df.columns]
df = df.drop(columns=drop_cols)
df['diagnosis'] = df['diagnosis'].map({'M':1, 'B':0})

top_features = ['area_worst', 'concave points_worst', 'concave points_mean',
                'radius_worst', 'perimeter_worst', 'perimeter_mean', 'concavity_mean',
                'area_mean', 'concavity_worst', 'radius_mean', 'area_se',
                'compactness_worst', 'texture_worst', 'texture_mean', 'radius_se']

X = df[top_features]
y = df['diagnosis']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ==========================
# Kaydedilmiş modeller ve scaler dizini
# ==========================
model_dir = os.path.join(os.getcwd(), "saved_models2")

# Scaler'ı yükle ve veriyi ölçekle
scaler = load(os.path.join(model_dir, "scaler_standard2.joblib"))
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modelleri yükle
log_reg = load(os.path.join(model_dir, "logistic_regression2.joblib"))
svm_model = load(os.path.join(model_dir, "svm_rbf2.joblib"))
rf_model = load(os.path.join(model_dir, "random_forest2.joblib"))
# ann_model = keras.models.load_model(os.path.join(model_dir, "ann_model2.h5"))

# ==========================
# Test seti üzerinde performans
# ==========================
models = {
    "Lojistik Regresyon": log_reg,
    "SVM": svm_model,
    "Random Forest": rf_model
}

print("\n--- Test Seti Sonuçları ---")
best_auc = 0
best_model_name = None
for name, model in models.items():
    if name != "ANN":
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:,1]
        # Çapraz doğrulama ile ROC-AUC
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
    else:
        y_pred = (model.predict(X_test_scaled) > 0.5).astype(int)
        y_proba = model.predict(X_test_scaled).ravel()
        # Keras için manuel ROC-AUC hesapla
        cv_scores = [roc_auc_score(y_train, model.predict(X_train_scaled).ravel())]
    
    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)
    print(f"{name} -> Accuracy: {acc:.4f}, ROC-AUC: {roc:.4f}, CV-ROC-AUC ort.: {np.mean(cv_scores):.4f}")
    
    if roc > best_auc:
        best_auc = roc
        best_model_name = name

print(f"\nEn iyi model: {best_model_name} (ROC-AUC: {best_auc:.4f})")
