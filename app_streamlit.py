# app.py
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import numpy as np
import streamlit as st

from joblib import load

# =======================
#  SAYFA KONFİGÜRASYONU
# =======================
st.set_page_config(
    page_title="Göğüs Kanseri Teşhis Sistemi | TÜBİTAK",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =======================
#  YARDIMCI SABİTLER
# =======================
APP_TITLE = "🎗️ Makine Öğrenmesi ile Göğüs Kanseri Teşhis Sistemi"
APP_SUBTITLE = (
    "Lütfen hastanın tümör ölçümlerine ilişkin **15 temel özelliği** aşağıdaki forma girin ve tahmini hesaplayın. "
    "Bu araç **klinik kararı desteklemek** içindir; tek başına tıbbi tanı koymaz."
)

# Özellikler (ANAHTAR, ETİKET). DİKKAT: Eğitimde kullanılan isimlerle **birebir aynı** olmalı!
# 'concave_points_mean' isminde BOŞLUK YOKTUR. (Yaygın bir yazım hatası düzeltilmiştir.)
FEATURES = [
    ("radius_mean", "Ortalama Yarıçap"),
    ("texture_mean", "Ortalama Doku"),
    ("perimeter_mean", "Ortalama Çevre Uzunluğu"),
    ("area_mean", "Ortalama Alan"),
    ("concavity_mean", "Ortalama Konkavlık"),
    ("concave_points_mean", "Ortalama Konkav Noktalar"),
    ("radius_se", "Yarıçap Standart Hatası"),
    ("area_se", "Alan Standart Hatası"),
    ("radius_worst", "En Kötü Yarıçap"),
    ("texture_worst", "En Kötü Doku"),
    ("perimeter_worst", "En Kötü Çevre Uzunluğu"),
    ("area_worst", "En Kötü Alan"),
    ("compactness_worst", "En Kötü Kompaktlık"),
    ("concavity_worst", "En Kötü Konkavlık"),
    ("concave_points_worst", "En Kötü Konkav Noktalar"),
]

# Kullanıcı girişi için makul aralıklar (isteğe bağlı—verinin tipik ölçekleri)
# Aralıklar veri setine göre güncellenebilir; bilinmiyorsa geniş bırakıldı.
INPUT_RANGES = {
    "radius_mean": (0.0, 40.0),
    "texture_mean": (0.0, 50.0),
    "perimeter_mean": (0.0, 300.0),
    "area_mean": (0.0, 4000.0),
    "concavity_mean": (0.0, 1.5),
    "concave_points_mean": (0.0, 1.0),
    "radius_se": (0.0, 10.0),
    "area_se": (0.0, 1000.0),
    "radius_worst": (0.0, 50.0),
    "texture_worst": (0.0, 60.0),
    "perimeter_worst": (0.0, 400.0),
    "area_worst": (0.0, 6000.0),
    "compactness_worst": (0.0, 1.5),
    "concavity_worst": (0.0, 2.0),
    "concave_points_worst": (0.0, 1.0),
}

# Varsayılan adım hassasiyeti
STEP = {
    "default": 0.0001,
    "int_like": 1.0
}

# =======================
#  MODEL & SCALER YÜKLEME
# =======================
@st.cache_resource(show_spinner=True)
def load_rf_model_and_scaler(model_dir: Path):
    """
    Model ve scaler'ı yükler, cache'ler.
    Dizin yapısı:
        models/
            random_forest2.joblib
            scaler_standard2.joblib
    """
    model_path = model_dir / "random_forest2.joblib"
    scaler_path = model_dir / "scaler_standard2.joblib"

    if not model_path.exists() or not scaler_path.exists():
        return None, None, {
            "exists": False,
            "model_path": str(model_path),
            "scaler_path": str(scaler_path)
        }

    try:
        rf_model = load(str(model_path))
        scaler = load(str(scaler_path))
        return rf_model, scaler, {"exists": True}
    except Exception as e:
        return None, None, {"exists": False, "error": str(e)}

# Çalışma dizini / model klasörü
CWD = Path(os.getcwd())
MODEL_DIR = CWD / "models"

rf_model, scaler, load_info = load_rf_model_and_scaler(MODEL_DIR)

# =======================
#  KENAR ÇUBUĞU (TÜBİTAK)
# =======================
with st.sidebar:
    st.title("Proje Bilgileri")
    st.info(
        "Bu web uygulaması, bir **TÜBİTAK** projesi kapsamında geliştirilmiştir. "
        "Amaç, makine öğrenmesi teknikleri kullanarak **göğüs kanseri teşhisinde** klinisyenlere **karar desteği** sunmaktır."
    )
    with st.expander("Kullanılan Model ve Teknolojiler", expanded=False):
        st.markdown(
            """
            - **Model:** Random Forest  
            - **Kütüphaneler:** Scikit-learn, NumPy, Joblib  
            - **Arayüz:** Streamlit  
            - **Ölçekleme:** Standard Scaler (eğitimle aynı olmalı)
            """
        )
    with st.expander("Önemli Notlar ve Etik", expanded=False):
        st.markdown(
            """
            - Bu uygulama **tanı koymaz**, hekim kararını **destekler**.  
            - Çıktılar, klinik bulgular ve görüntüleme sonuçlarıyla **birlikte** değerlendirilmelidir.  
            - **KVKK** kapsamında kişisel veriler **saklanmaz**; girilen veriler sadece anlık hesaplama için kullanılır.  
            - Modelin performansı eğitim verisi, kurulum ve kullanım koşullarına bağlı olarak değişebilir.
            """
        )
    st.caption("© 2025 — Tüm Hakları Saklıdır.")

# =======================
#  ANA ARAYÜZ
# =======================
st.title(APP_TITLE)
st.markdown(APP_SUBTITLE)

# Model / scaler bulunamadıysa net uyarı
if not load_info.get("exists", False) or rf_model is None or scaler is None:
    st.error("Model veya scaler yüklenemedi. Lütfen aşağıdaki yolları kontrol edin ve dosyaların mevcut olduğundan emin olun.")
    with st.expander("Kurulum Kontrol Listesi", expanded=True):
        st.markdown(
            f"""
            - Model dizini: `{MODEL_DIR}`  
            - Beklenen model: `{MODEL_DIR / 'random_forest2.joblib'}`  
            - Beklenen scaler: `{MODEL_DIR / 'scaler_standard2.joblib'}`  
            - Dosya adları eğitim sırasında kaydedilenlerle **birebir aynı** olmalı.  
            - Python ortamında `scikit-learn`, `joblib`, `streamlit`, `numpy` paketleri kurulu olmalı.
            """
        )
else:
    # === Form ===
    with st.form("prediction_form", clear_on_submit=False):
        st.subheader("Hasta Veri Girişi")

        # İki sütunlu düzen
        col1, col2 = st.columns(2)
        inputs = {}

        for i, (fname, flabel) in enumerate(FEATURES):
            min_v, max_v = INPUT_RANGES.get(fname, (0.0, 1e6))
            step = STEP["default"]

            # Birkaç özelliğe daha büyük step (görsel kolaylık) tanımlayalım
            if fname.endswith("_mean") or fname.endswith("_worst") or fname.endswith("_se"):
                step = STEP["default"]

            # Sütuna dağıt
            container = col1 if i < len(FEATURES) / 2 else col2
            with container:
                inputs[fname] = st.number_input(
                    label=flabel,
                    min_value=float(min_v),
                    max_value=float(max_v),
                    value=float(min_v),
                    step=float(step),
                    format="%.4f",
                    help=f"{flabel} değerini giriniz. Aralık: [{min_v}, {max_v}]"
                )

        # Eşik ayarı (opsiyonel)
        st.divider()
        th_col1, th_col2 = st.columns([2, 1])
        with th_col1:
            st.caption("Modelin pozitif sınıf (kötü huylu) karar eşiğini ayarlayabilirsiniz.")
        with th_col2:
            threshold = st.number_input(
                label="Karar Eşiği",
                min_value=0.05, max_value=0.95, value=0.50, step=0.01, format="%.2f",
                help="Tahmin olasılığı bu eşiğin üzerindeyse sonuç 'Kötü Huylu' kabul edilir."
            )

        submitted = st.form_submit_button("Teşhis Sonucunu Hesapla", use_container_width=True)

    # === Tahmin ===
    if submitted:
        try:
            # 1) Girdileri doğru sırayla diziye dönüştür
            x_list = [float(inputs[f]) for f, _ in FEATURES]
            X = np.array(x_list, dtype=np.float32).reshape(1, -1)

            # 2) Ölçekle
            Xs = scaler.transform(X)

            # 3) Tahmin
            proba = float(rf_model.predict_proba(Xs)[0][1])
            pred = 1 if proba >= threshold else 0

            # 4) Sonuç Sunumu (baskı/PDF için uygun)
            st.subheader("Teşhis Sonucu")
            st.progress(int(round(proba * 100)), text=f"Kötü huylu olasılığı: {proba:.2%}")

            if pred == 1:
                st.error(
                    f"**Sonuç: Kötü Huylu (Kanserli) — Olasılık: {proba:.2%}**\n\n"
                    f"Seçili karar eşiği: **{threshold:.2f}**"
                )
            else:
                st.success(
                    f"**Sonuç: İyi Huylu (Kansersiz) — Olasılık: {(1.0 - proba):.2%}**\n\n"
                    f"Seçili karar eşiği: **{threshold:.2f}**"
                )

            with st.expander("Girilen Değerlerin Özeti"):
                # Tablo gibi görünmesi için Markdown
                md_lines = ["| Özellik | Değer |", "|---|---|"]
                for key, label in FEATURES:
                    md_lines.append(f"| {label} | {inputs[key]:.4f} |")
                st.markdown("\n".join(md_lines))

            st.info(
                "Bu çıktı **klinik kararı desteklemek** amacıyla sunulmuştur ve tek başına tanı için **yeterli değildir**. "
                "Klinik bulgular ve görüntüleme sonuçlarıyla birlikte değerlendirilmelidir."
            )

        except Exception as e:
            st.exception(e)
            st.error("Tahmin sırasında beklenmeyen bir hata oluştu. Lütfen girdileri ve model dosyalarını kontrol edin.")

# =======================
#  ALT BİLGİ (BASKI DOSTU)
# =======================
st.markdown("---")
st.caption(
    "Sürüm: 1.0.0 • Geliştirme: 2025 • "
    "Bu sayfayı PDF'e aktarmak için sağ üstten tarayıcı **Yazdır** (Ctrl/Cmd+P) ile **PDF olarak kaydedin**."
)
