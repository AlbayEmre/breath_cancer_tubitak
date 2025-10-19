# ÖZELLİK SEÇİMİ İLE GÖĞÜS KANSERİ TESPİTİ
# 15 ÖNEMLİ ÖZELLİK KULLANILDI

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
# import tensorflow as tf
# from tensorflow import keras
from joblib import dump, load
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# ==========================
# Veri seti ve önemli özellikler
# ==========================
data_path = "data.csv"  # veri yolunu belirt
df = pd.read_csv(data_path)

drop_cols = [c for c in ['id', 'Unnamed: 32', 'Unnamed:32'] if c in df.columns]
df = df.drop(columns=drop_cols)
df['diagnosis'] = df['diagnosis'].map({'M':1, 'B':0})

# Daha önce seçilen en önemli 15 özellik
top_features = ['area_worst', 'concave points_worst', 'concave points_mean',
                'radius_worst', 'perimeter_worst', 'perimeter_mean', 'concavity_mean',
                'area_mean', 'concavity_worst', 'radius_mean', 'area_se',
                'compactness_worst', 'texture_worst', 'texture_mean', 'radius_se']

X = df[top_features]
y = df['diagnosis']

# Eğitim/Test seti
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Ölçekleme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================
# Modelleri eğit
# ==========================
# 1) Lojistik Regresyon
log_reg = LogisticRegression(max_iter=500)
log_reg.fit(X_train_scaled, y_train)

# 2) SVM
svm_model = SVC(kernel='rbf', probability=True)
svm_model.fit(X_train_scaled, y_train)

# 3) Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# 4) ANN
# ann_model = keras.Sequential([
#     keras.layers.Dense(16, activation='relu', input_shape=(X_train_scaled.shape[1],)),
#     keras.layers.Dropout(0.2),
#     keras.layers.Dense(8, activation='relu'),
#     keras.layers.Dense(1, activation='sigmoid')
# ])
# ann_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# ann_model.fit(X_train_scaled, y_train, epochs=50, batch_size=16, verbose=0,
#               validation_data=(X_test_scaled, y_test))

# ==========================
# Test setinde performans
# ==========================
models = {
    "Lojistik Regresyon": log_reg,
    "SVM": svm_model,
    "Random Forest": rf_model
}

print("\n--- Test Seti Sonuçları ---")
for name, model in models.items():
    if name != "ANN":
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:,1]
    else:
        y_pred = (model.predict(X_test_scaled) > 0.5).astype(int)
        y_proba = model.predict(X_test_scaled).ravel()
    
    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)
    print(f"{name} -> Accuracy: {acc:.4f}, ROC-AUC: {roc:.4f}")

# ==========================
# ROC eğrilerini çiz
# ==========================
plt.figure(figsize=(8,6))

def plot_roc(y_true, y_pred_proba, label):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'{label} (AUC = {roc_auc:.3f})')

for name, model in models.items():
    if name != "ANN":
        plot_roc(y_test, model.predict_proba(X_test_scaled)[:,1], name)
    else:
        plot_roc(y_test, model.predict(X_test_scaled).ravel(), name)

plt.plot([0,1], [0,1], color='navy', lw=1, linestyle='--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Eğrileri Karşılaştırması (15 Önemli Özellik)')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# ==========================
# Modelleri ve scaler'ı kaydet (çalışma dizinine)
# ==========================
model_dir = os.path.join(os.getcwd(), "models")
os.makedirs(model_dir, exist_ok=True)

# Modeller
dump(log_reg, os.path.join(model_dir, "logistic_regression2.joblib"))
dump(svm_model, os.path.join(model_dir, "svm_rbf2.joblib"))
dump(rf_model, os.path.join(model_dir, "random_forest2.joblib"))
# ann_model.save(os.path.join(model_dir, "ann_model2.h5"))

# Scaler
dump(scaler, os.path.join(model_dir, "scaler_standard2.joblib"))

print("Tüm modeller ve scaler kaydedildi:", model_dir)
