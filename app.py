import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st

from datetime import datetime

start=datetime(2010,1,1)
end=datetime.now()

st.title("Stock trend Prediction")

user_input=st.text_input("Enter Stock Ticker (eg. AAPL, TSLA):","SBIN.NS")
df=yf.download(user_input,start,end)

#Describe data
st.subheader("Data from 2010 till now ")
st.write(df.describe())

#Visualizations
st.subheader("Closing Price vs Time Chart")
fig=plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

#Visualization with 100MA
st.subheader("Closing Price vs Time Chart with 100MA")
ma100=df.Close.rolling(100).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100,"green",label="100 MA")
plt.plot(df.Close,label="Price")
plt.legend()
st.pyplot(fig)

#Visualisation with 100MA and 200MA
st.subheader("Closing Price vs Time Chart with 100MA and 200MA")
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100,"green",label="100 MA")
plt.plot(ma200,"pink",label="200 MA")
plt.plot(df.Close,label="Price")
plt.legend()
st.pyplot(fig)

#Splitting into training and testing
data_training=pd.DataFrame(df["Close"][0:int(len(df)*0.70)])
data_test=pd.DataFrame(df["Close"][int(len(df)*0.70) : int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scalar=MinMaxScaler(feature_range=(0,1))
#data_training_array=scalar.fit_transform(data_training)

#Load my model
model=load_model("stocks_model.h5")

#Testing part
past_100_days=data_training.tail(100)
final_df=past_100_days.append(data_test,ignore_index=True)
input_data=scalar.fit_transform(final_df)

x_test=[]
y_test=[]
for i in  range(100,input_data.shape[0]):
  x_test.append(input_data[i-100 :i])
  y_test.append(input_data[i,0])

x_test,y_test= np.array(x_test), np.array(y_test)

#Making predictions
y_pred=model.predict(x_test)

scale=scalar.scale_
scale_factor=1/scale[0]
y_pred=y_pred*scale_factor
y_test=y_test*scale_factor

#Final Graph
st.subheader("Predictions vs Orignal")
fig2=plt.figure(figsize=(12,6))
plt.plot(y_test,"blue",label="Orignal Price")
plt.plot(y_pred,"red",label="Predicted Price")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
st.pyplot(fig2)
