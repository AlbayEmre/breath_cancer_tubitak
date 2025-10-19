import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Veri setini oku
data_path = "data.csv"  # kendi yolunu belirt
df = pd.read_csv(data_path)

# Gereksiz sütunları çıkar
drop_cols = [c for c in ['id', 'Unnamed: 32', 'Unnamed:32'] if c in df.columns]
df = df.drop(columns=drop_cols)

# Hedef değişkeni sayısallaştır
df['diagnosis'] = df['diagnosis'].map({'M':1, 'B':0})

# 1️⃣ Korelasyon matrisi ile yüksek ilişkili değişkenleri kontrol et
corr_matrix = df.corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
plt.title("Korelasyon Matrisi")
plt.show()

# 2️⃣ Random Forest ile özellik önemi
X = df.drop(columns=['diagnosis'])
y = df['diagnosis']

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

print("Özellik Sıralaması (önem derecesine göre):")
for f in range(X.shape[1]):
    print(f"{f+1}. {X.columns[indices[f]]} ({importances[indices[f]]:.4f})")

# 3️⃣ Önemli özellikleri seç (örnek: 15 üst sıradaki)
top_features = X.columns[indices[:15]]
print("\nSeçilen en önemli özellikler:")
print(top_features.tolist())

# Artık bu top_features ile modellerini yeniden eğitebilirsin:
X_selected = X[top_features]
