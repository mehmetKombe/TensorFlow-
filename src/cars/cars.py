from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
import seaborn as sbn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout

# .env dosyasını yükle
load_dotenv()
excel_path = os.getenv("EXCEL_PATH")

# Veriyi oku
dataFrame = pd.read_excel(excel_path)

# Dağılım grafikleri
sbn.displot(dataFrame["price"])
sbn.countplot(data=dataFrame, x="year")

# Fiyatı en yüksek 131 aracı çıkar
dataFrame = dataFrame.sort_values("price", ascending=False).iloc[131:]
# 1970 verilerini temizle
dataFrame = dataFrame[dataFrame["year"] != 1970]

# transmission'ı one-hot encode et
dataFrame = pd.get_dummies(dataFrame, columns=["transmission"], drop_first=True)

# Hedef ve bağımsız değişkenleri ayır
y = dataFrame["price"].values
x = dataFrame.drop("price", axis=1).values

# Eğitim ve test verilerini ayır
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=10)

# Ölçeklendirme
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Model
model = Sequential()
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.2))  
model.add(Dense(256, activation="relu"))
# model.add(Dropout(0.4)) 
model.add(Dense(128, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(1))

model.compile(optimizer="adam", loss="mse")

early_stop = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=25, restore_best_weights=True)

# Eğit
model.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), batch_size=250, epochs=500, verbose=1)

# Tahmin
tahminler = model.predict(x_test)

# R2 skoru
r2 = r2_score(y_test, tahminler)
print("R² (Doğruluk Skoru):", r2)

# Grafik
plt.figure(figsize=(10, 6))
plt.scatter(y_test, tahminler)
plt.plot(y_test, y_test, "r--")
plt.xlabel("Gerçek Fiyat")
plt.ylabel("Tahmin Edilen Fiyat")
plt.title("Gerçek vs Tahmin Edilen Araç Fiyatları")
plt.show()
