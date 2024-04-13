import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
import matplotlib.pyplot as plt

def main()->None:
    
    df = pd.read_csv("Data/features_3_sec.csv")

    class_encod=df.iloc[:,-1]
    converter=LabelEncoder()
    y=converter.fit_transform(class_encod)

    df=df.drop(labels="filename",axis=1)


    df_new = pd.read_csv("myfeatures_3_sec.csv")
    df_new=df_new.drop(labels="filename",axis=1)


    fit = StandardScaler()
    fit.fit_transform(np.array(df.iloc[:,:-1],dtype=float))

    model = keras.models.load_model("good.h5")

    sample = df_new.iloc[0,:-1] 
    sample_scaled = fit.transform(np.array(sample).reshape(1,-1))

    predicted_class = np.argmax(model.predict(sample_scaled))
    predicted_name = converter.inverse_transform([predicted_class])[0]

    print(predicted_name)

if __name__ == "__main__":
    main()