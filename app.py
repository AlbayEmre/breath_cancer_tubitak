from flask import Flask, render_template, request
import numpy as np
import os
from tensorflow import keras
from joblib import load

app = Flask(__name__)

# Model ve scaler yükle
model_dir = os.path.join(os.getcwd(), "saved_models2")
ann_model = keras.models.load_model(os.path.join(model_dir, "ann_model2.h5"))
scaler = load(os.path.join(model_dir, "scaler_standard2.joblib"))

# 15 önemli özellik ve açıklamaları
features = [
    ("area_worst", "En kötü tümör alanı"),
    ("concave points_worst", "En kötü konkav nokta sayısı"),
    ("concave points_mean", "Ortalama konkav nokta sayısı"),
    ("radius_worst", "En kötü yarıçap"),
    ("perimeter_worst", "En kötü çevre uzunluğu"),
    ("perimeter_mean", "Ortalama çevre uzunluğu"),
    ("concavity_mean", "Ortalama konvekslik"),
    ("area_mean", "Ortalama tümör alanı"),
    ("concavity_worst", "En kötü konvekslik"),
    ("radius_mean", "Ortalama yarıçap"),
    ("area_se", "Alan standart hatası"),
    ("compactness_worst", "En kötü kompaktlık"),
    ("texture_worst", "En kötü doku"),
    ("texture_mean", "Ortalama doku"),
    ("radius_se", "Yarıçap standart hatası")
]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error_msg = None
    if request.method == "POST":
        try:
            # Kullanıcıdan gelen değerleri al
            input_values = []
            for f, _ in features:
                val = request.form[f]
                input_values.append(float(val))  # validasyon
            X_input = np.array(input_values).reshape(1, -1)
            
            # Önceden eğitilmiş scaler ile dönüştür
            X_scaled = scaler.transform(X_input)
            y_pred = (ann_model.predict(X_scaled) > 0.5).astype(int)[0][0]
            
            prediction = "Kötü Huylu (Kanserli)" if y_pred == 1 else "İyi Huylu (Kansersiz)"
        except ValueError:
            error_msg = "Lütfen tüm değerleri geçerli sayılar olarak giriniz."
    
    return render_template("index.html", features=features, prediction=prediction, error_msg=error_msg)

if __name__ == "__main__":
    app.run(debug=True)
