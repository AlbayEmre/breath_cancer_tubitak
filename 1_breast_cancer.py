import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import dump

# ==========================
# Veri Hazırlama (İP 1)
# ==========================

# 1) Veri setini oku
data_path = "data.csv"  # kendi yolunu belirt
df = pd.read_csv(data_path)
print("Ham veri boyutu:", df.shape)

# 2) Gereksiz sütunları çıkar
drop_cols = [c for c in ['id', 'Unnamed: 32', 'Unnamed:32'] if c in df.columns]
df = df.drop(columns=drop_cols)

# 3) Hedef değişkeni sayısallaştır (M:1, B:0)
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0}).astype(int)

# 4) Eksik değer kontrolü
missing = df.isna().sum()
if missing.sum() > 0:
    print("Eksik değerler var, ortalama ile doldurulacak")
    df = df.fillna(df.mean())
else:
    print("Eksik değer yok.")

# 5) X ve y ayrımı
X = df.drop(columns=['diagnosis'])
y = df['diagnosis']

# 6) Eğitim/Test seti ayır (stratify ile dengeli dağılım)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print("Eğitim boyutu:", X_train.shape, " Test boyutu:", X_test.shape)

# 7) Ölçekleme (yalnızca eğitimde fit)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 8) Sonuçları kaydetme (opsiyonel)
out_dir = "/mnt/data/preprocessed_outputs"
os.makedirs(out_dir, exist_ok=True)

np.save(os.path.join(out_dir, "X_train_scaled.npy"), X_train_scaled)
np.save(os.path.join(out_dir, "X_test_scaled.npy"), X_test_scaled)
np.save(os.path.join(out_dir, "y_train.npy"), y_train.values)
np.save(os.path.join(out_dir, "y_test.npy"), y_test.values)
dump(scaler, os.path.join(out_dir, "scaler_standard.joblib"))

print("Ölçekleyici ve numpy dosyaları kaydedildi:", out_dir)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Kaydedilmiş ölçekli verileri yükle (ya da yukarıda tanımladıklarını kullan)
X_train = np.load("/mnt/data/preprocessed_outputs/X_train_scaled.npy")
X_test = np.load("/mnt/data/preprocessed_outputs/X_test_scaled.npy")
y_train = np.load("/mnt/data/preprocessed_outputs/y_train.npy")
y_test = np.load("/mnt/data/preprocessed_outputs/y_test.npy")

# ----------------------------
# 1) Lojistik Regresyon
# ----------------------------
log_reg = LogisticRegression(max_iter=500)
log_reg.fit(X_train, y_train)
y_pred_lr = log_reg.predict(X_test)
print("Lojistik Regresyon Accuracy:", accuracy_score(y_test, y_pred_lr))
print("ROC-AUC:", roc_auc_score(y_test, log_reg.predict_proba(X_test)[:,1]))

# ----------------------------
# 2) SVM (RBF kernel)
# ----------------------------
svm_model = SVC(kernel='rbf', probability=True)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
print("\nSVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print("ROC-AUC:", roc_auc_score(y_test, svm_model.predict_proba(X_test)[:,1]))

# ----------------------------
# 3) Random Forest
# ----------------------------
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print("\nRandom Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("ROC-AUC:", roc_auc_score(y_test, rf_model.predict_proba(X_test)[:,1]))

# ----------------------------
# 4) Basit ANN (Keras)
# ----------------------------
ann_model = keras.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

ann_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
ann_model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0,
              validation_data=(X_test, y_test))

ann_eval = ann_model.evaluate(X_test, y_test, verbose=0)
print("\nANN Test Loss, Accuracy:", ann_eval)
y_pred_ann = (ann_model.predict(X_test) > 0.5).astype(int)
print("ANN ROC-AUC:", roc_auc_score(y_test, ann_model.predict(X_test)))

# ----------------------------
# Çapraz doğrulama örneği (lojistik için)
# ----------------------------
cv_scores = cross_val_score(log_reg, X_train, y_train, cv=10, scoring='roc_auc')
print("\nLojistik Regresyon 10-Fold ROC-AUC ortalama:", np.mean(cv_scores))

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# ROC çizim fonksiyonu
def plot_roc(y_true, y_pred_proba, label):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'{label} (AUC = {roc_auc:.3f})')

plt.figure(figsize=(8, 6))

# 1) Lojistik Regresyon
plot_roc(y_test, log_reg.predict_proba(X_test)[:,1], 'Lojistik Regresyon')

# 2) SVM
plot_roc(y_test, svm_model.predict_proba(X_test)[:,1], 'SVM')

# 3) Random Forest
plot_roc(y_test, rf_model.predict_proba(X_test)[:,1], 'Random Forest')

# 4) ANN
plot_roc(y_test, ann_model.predict(X_test).ravel(), 'ANN')

# Rastgele tahmin referansı
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Eğrileri Karşılaştırması')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

from joblib import dump, load
import os

# ==========================
# Modelleri kaydetme (çalışma dizinine)
# ==========================
model_dir = os.path.join(os.getcwd(), "saved_models")
os.makedirs(model_dir, exist_ok=True)

# Lojistik Regresyon
dump(log_reg, os.path.join(model_dir, "logistic_regression.joblib"))

# SVM
dump(svm_model, os.path.join(model_dir, "svm_rbf.joblib"))

# Random Forest
dump(rf_model, os.path.join(model_dir, "random_forest.joblib"))

# ANN (Keras modeli)
ann_model.save(os.path.join(model_dir, "ann_model.h5"))

print("Tüm modeller kaydedildi:", model_dir)

# ==========================
# Test veri seti üzerinde yükleyip deneme
# ==========================
# Klasik modelleri yükle
log_reg_loaded = load(os.path.join(model_dir, "logistic_regression.joblib"))
svm_loaded = load(os.path.join(model_dir, "svm_rbf.joblib"))
rf_loaded = load(os.path.join(model_dir, "random_forest.joblib"))

# Keras modeli yükle
ann_loaded = keras.models.load_model(os.path.join(model_dir, "ann_model.h5"))

# Test veri seti üzerinde tahmin
print("\n--- Test Seti Sonuçları ---")
for name, model in zip(
    ["Lojistik Regresyon", "SVM", "Random Forest", "ANN"],
    [log_reg_loaded, svm_loaded, rf_loaded, ann_loaded]
):
    if name != "ANN":
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:,1]
    else:
        y_pred = (model.predict(X_test) > 0.5).astype(int)
        y_proba = model.predict(X_test).ravel()
    
    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)
    print(f"{name} -> Accuracy: {acc:.4f}, ROC-AUC: {roc:.4f}")