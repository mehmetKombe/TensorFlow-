import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# 1. Veri yükle
df = pd.read_csv("winequality-red.csv")
df = df[df['fixed acidity'] <= 12]

# 2. 3 sınıflı hedef sütunu oluştur
def kalite_etiketle(q):
    if q >= 7:
        return 2  # iyi
    elif q >= 5:
        return 1  # orta
    else:
        return 0  # kötü

df['quality_label'] = df['quality'].apply(kalite_etiketle)

# 3. Girdi ve hedef
X = df.drop(columns=['quality', 'quality_label'])
y = df['quality_label']

# 4. Train/Test böl
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 5. StandardScaler uygula
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
joblib.dump(scaler, "scaler.pkl")

# 6. Model oluştur
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')  # 3 sınıf
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 7. Eğit
history = model.fit(X_train, y_train, epochs=100, validation_split=0.2, verbose=0)

# 8. Tahmin yap
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)  # 3 sınıfa göre

# 9. Değerlendirme
acc = accuracy_score(y_test, y_pred)
print(f"Doğruluk (accuracy): {acc:.4f}\n")
print("🔍 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred))

# 10. Doğruluk grafiği
plt.figure(figsize=(6, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
plt.plot([0, 2], [0, 2], 'r--')
plt.title("Gerçek vs Tahmin Edilen Sınıf")
plt.xlabel("Gerçek Değer")
plt.ylabel("Tahmin Edilen Değer")
plt.grid(True)
plt.show()

# 11. Modeli kaydet
model.save("sarap_modeli.h5")
