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





load_dotenv()  # .env dosyasını yükler
excel_path = os.getenv("EXCEL_PATH")  # .env içinden değeri alır

dataFrame = pd.read_excel(excel_path)

sbn.displot(dataFrame["price"])
sbn.countplot(dataFrame["year"])
# plt.show()


#aşağıdaki kod en yüksek 131 fiyatlı arabaları atlar (verfi temizliği için bunu yaparız standart sapmayı fazla etkilememesi için)
ayiklanmisData=dataFrame.sort_values("price",ascending=False).iloc[131:]
dataFrame=ayiklanmisData
dataFrame=dataFrame[dataFrame.year!=1970]


# print(dataFrame.groupby("year").mean()["price"])

dataFrame=dataFrame.drop("transmission",axis=1)

dataFrame.groupby("year").mean()["price"]

y=dataFrame["price"].values
x=dataFrame.drop("price",axis=1).values

x_train, x_test, y_train, y_test =train_test_split(x,y,test_size=0.3,random_state=10)
scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)
print(x_train.shape)

model=Sequential()

model.add(Dense(64, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(64, activation="relu"))

model.add(Dense(1))

model.compile(optimizer="adam",loss="mse")

model.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), batch_size=250, epochs=500)

# Test verileri ile tahmin yap
tahminler = model.predict(x_test)

# R^2 doğruluk skoru
r2 = r2_score(y_test, tahminler)
print("R² (Doğruluk Skoru):", r2)


# Tahmin vs Gerçek Fiyat grafiği
plt.figure(figsize=(10,6))
plt.scatter(y_test, tahminler)
plt.plot(y_test, y_test, "r--")  # Doğru tahmin çizgisi
plt.xlabel("Gerçek Fiyat")
plt.ylabel("Tahmin Edilen Fiyat")
plt.title("Gerçek vs Tahmin Edilen Araç Fiyatları")
plt.show()