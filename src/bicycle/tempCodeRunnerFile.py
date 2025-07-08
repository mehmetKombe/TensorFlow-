#source .venv/bin/activate
import seaborn as sbn
import matplotlib.pylab as plt 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score
from tensorflow.keras.models import load_model



dataFrame=pd.read_excel("C:/Users/mehmetkombe/Desktop/Staj Anka/Bisiklet_Fiyat_Tahmin/bisiklet_verisi.xlsx")

# sbn.pairplot(dataFrame)

#y= wx+b
#y-> label(tahmin edeceğimiz değer)
y=dataFrame["fiyat_tl"].values

#x-> feature(özellikler) anlamına gelir ve ne kadar özelliklerimiz varsa buraya o kadar verilerimizi yazmamız lazım 
x=dataFrame[["uretim_yili","binis_km"]].values

x_train,x_test, y_train, y_test=train_test_split(x,y,test_size=0.33,random_state=22)
x_train.shape
print(x_train.shape)

#scaling yani verdiğimiz değerleri 0-1 arsına alarak daha iyi çalışmasını sağlar MinMaxScaler fonksiyonu
scaler=MinMaxScaler()

#aşağıdaki kod verimizdeki max değeri bulur ve buna 1 min değeri bulur ve buna 0 verir 
scaler.fit(x_train)

# scaler.fit(...) ile öğrenilen min-max değerlerine göre
# x_train ve x_test verileri 0 ile 1 arasına dönüştürülür
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

#model oluşturulma işlemi
model=Sequential()

#Aşağıya yazdığımız kodlar ne kadar hidden layer(gizli katman) istersem o kadar kodu aynı şekil yazıyorum 
model.add(Dense(8,activation="relu"))
model.add(Dense(8,activation="relu"))
model.add(Dense(8,activation="relu"))

#Çıktı nöronu
model.add(Dense(1))

#öğrenme oranı optimizer ve hata oranı loss
model.compile(optimizer="adam", loss="mse")

#model eğitim başlatma komutları işte neyi eğiteceğimiz fln 
model.fit(x_train,y_train,epochs=500)

loss = model.evaluate(x_test, y_test)
print("Test MSE:", loss)

tahminler = model.predict(x_test)


#doğruluk oranı hesaplama
r2 = r2_score(y_test, tahminler)
print("R^2 (doğruluk skoru):", r2)

#Tahmin grafiği
plt.figure(figsize=(10,6))
plt.scatter(y_test, tahminler)
plt.plot(y_test, y_test, "r--")  # Doğru tahmin çizgisi
plt.xlabel("Gerçek Fiyat")
plt.ylabel("Tahmin Edilen Fiyat")
plt.title("Gerçek vs Tahmin Edilen Bisiklet Fiyatları")
plt.show()


yeniBisikletOzellikleri=[[3250,3251]]
yeniBisikletOzellikleri=scaler.transform(yeniBisikletOzellikleri)
#kaydetme
model.save("bisiklet_modeli.h5")
sonradanCagirilanModel=load_model("bisiklet_modeli.h5")
sonradanCagirilanModel.predict(yeniBisikletOzellikleri)