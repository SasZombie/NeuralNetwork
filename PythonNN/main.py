import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pickle


df = pd.read_csv("Data/features_3_sec.csv")

class_encod=df.iloc[:,-1]
converter=LabelEncoder()
y=converter.fit_transform(class_encod)

df=df.drop(labels="filename",axis=1)

fit=StandardScaler()
X=fit.fit_transform(np.array(df.iloc[:,:-1],dtype=float))
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)



df_new = pd.read_csv("myfeatures_3_sec.csv")
df_new=df_new.drop(labels="filename",axis=1)


fit = StandardScaler()
X1=fit.fit_transform(np.array(df.iloc[:,:-1],dtype=float))



def train_model(model,epochs,optimizer):
    batch_size=256
    model.compile(optimizer=optimizer,loss='sparse_categorical_crossentropy',metrics='accuracy')
    return model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=epochs,batch_size=batch_size)


model=tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(X.shape[1],)),
    tf.keras.layers.Dropout(0.2),
    
    tf.keras.layers.Dense(512,activation='relu'),
    keras.layers.Dropout(0.2),
    
    tf.keras.layers.Dense(256,activation='relu'),
    tf.keras.layers.Dropout(0.2),
    
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dropout(0.2),
    
    tf.keras.layers.Dense(64,activation='relu'),
    tf.keras.layers.Dropout(0.2),
    
    tf.keras.layers.Dense(32,activation='relu'),
    tf.keras.layers.Dropout(0.2),
    
    tf.keras.layers.Dense(10,activation='softmax'),
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.000146)
model.compile(optimizer=optimizer,
             loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.summary()
model_history=train_model(model=model, epochs=10, optimizer='adam')
# model.save("good.h5")
# tf.keras.models.save_model(model_history, 'model_history_tf', save_format='tf')

with open('model_history.pkl', 'wb') as file:
    pickle.dump(model_history.history, file)

sample = df_new.iloc[0,:-1] 
sample_scaled = fit.transform(np.array(sample).reshape(1,-1))

predicted_class = np.argmax(model.predict(sample_scaled))
predicted_name = converter.inverse_transform([predicted_class])[0]

print(predicted_name)