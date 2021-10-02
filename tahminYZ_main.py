# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 19:06:46 2021

@author: saUzu
"""

#%% Gerekli Kütüphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%% Dolar/Lira ve eğitim verisini hazırlama
#hamVeriler = pd.read_csv('02-21_egitim_verileri.csv')
        # bitcoin için değişkenler ve değerler
hamVeriler = pd.read_csv('Bitcoin_USD_ogrenme.csv')
hamVeriler['Price'] = hamVeriler['Price'].str.replace(',','')
hamVeriler['Open'] = hamVeriler['Open'].str.replace(',','')
hamVeriler['High'] = hamVeriler['High'].str.replace(',','')
hamVeriler['Low'] = hamVeriler['Low'].str.replace(',','')
        # bitcoin için bitiş
hamVeriler['Date'] = pd.to_datetime(hamVeriler['Date'])
hamVeriler = hamVeriler.sort_values(by='Date')
egitimVerisi = hamVeriler.iloc[:, 1:2].values
    
#%% Eğitim verisini ölçeklendirme
from sklearn.preprocessing import MinMaxScaler
olcek = MinMaxScaler(feature_range=(0, 1))
olcekli_egitimVerisi = olcek.fit_transform(egitimVerisi)   

#%% Eğitim verilerini 60'a böldüm. Her 60 günde bir tahmin yaparak öğrenme işlemini gerçekleştirecek
X_egitim = []
y_egitim = []
for i in range(60, 4080):
    X_egitim.append(olcekli_egitimVerisi[i-60:i, 0])
    y_egitim.append(olcekli_egitimVerisi[i, 0])
X_egitim, y_egitim = np.array(X_egitim), np.array(y_egitim)

#%% Eğitim verilerini yeniden şekillendirme işlemi
X_egitim = np.reshape(X_egitim, (X_egitim.shape[0], X_egitim.shape[1], 1))

#%% LSTM için keras kütüphanesi
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#%% RNN oluşturma
gerileme = Sequential()

#%% LSTM'nin ilk katmanı
gerileme.add(LSTM(units = 100, return_sequences = True, input_shape = (X_egitim.shape[1], 1)))
gerileme.add(Dropout(0.2))

#%% LSTM'nin ikinci katmanı
gerileme.add(LSTM(units = 100, return_sequences = True))
gerileme.add(Dropout(0.2))

#%% LSTM'nin üçüncü katmanı
gerileme.add(LSTM(units = 100, return_sequences = True))
gerileme.add(Dropout(0.2))

#%% LSTM'nin dördüncü katmanı
gerileme.add(LSTM(units = 100))
gerileme.add(Dropout(0.2))

#%% LSTM'nin Çıkış katmanı
gerileme.add(Dense(units = 1))

#%% RNN'yi çalıştırma
gerileme.compile(optimizer = 'adam', loss = 'mean_squared_error')

#%% RNN'yi eğitim verileri ile uyuşması işlemi
gerileme.fit(X_egitim, y_egitim, epochs = 100, batch_size = 100)

#%% Gerçek veriler
#denemeVerileri = pd.read_csv('deneme_verileri2.csv')

        # bitcoin için değerler ve değişkenle
denemeVerileri = pd.read_csv('Bitcoin_USD_deneme.csv')
denemeVerileri['Price'] = denemeVerileri['Price'].str.replace(',','')
denemeVerileri['Open'] = denemeVerileri['Open'].str.replace(',','')
denemeVerileri['High'] = denemeVerileri['High'].str.replace(',','')
denemeVerileri['Low'] = denemeVerileri['Low'].str.replace(',','')
        # bitcoin için bitiş

denemeVerileri['Date'] = pd.to_datetime(denemeVerileri['Date'])
denemeVerileri = denemeVerileri.sort_values(by='Date')
gercekVeriler = denemeVerileri.iloc[:,1:2].values

        # bitcoin değerleri için
gercekVeriler = np.array(gercekVeriler, dtype=np.float32)

#%% Tahmini verileri alma
tumVeriler = pd.concat((hamVeriler['Price'], denemeVerileri['Price']), axis = 0)
girisler = tumVeriler[len(tumVeriler)-len(denemeVerileri)-60:].values
girisler = girisler.reshape(-1, 1)
girisler = olcek.transform(girisler)

X_deneme = []
for i in range(60,320):
    X_deneme.append(girisler[i-60:i,0])
X_deneme = np.array(X_deneme)
X_deneme = np.reshape(X_deneme,(X_deneme.shape[0],X_deneme.shape[1],1))
tahminiDegerler = gerileme.predict(X_deneme)
tahminiDegerler = olcek.inverse_transform(tahminiDegerler)

#%% Plot ile görselleştirme
plt.plot(gercekVeriler, color='red', label='Gerçek Veriler')
plt.plot(tahminiDegerler, color='blue', label='Tahmin Edilen Değerler')
        # bitcoin için grafikler
plt.title('Bitcoin/USD Kur tahmini')
plt.ylabel('Bitcoin/USD')
        # bitcoin için bitiş
#plt.title('Dolar/TL Kur tahmini')
#plt.ylabel('Dolar/TL')
plt.xlabel('Tarih')
plt.legend()
plt.show()
