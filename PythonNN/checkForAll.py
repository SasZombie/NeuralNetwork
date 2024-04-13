import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pickle
from tensorflow import keras
import matplotlib.pyplot as plt



def Validation_plot(history)->None:
    print("Validation Accuracy", max(history["val_accuracy"]))
    pd.DataFrame(history).plot(figsize=(12, 6))
    plt.show()

def main()->None:
    
    df = pd.read_csv("Data/features_3_sec.csv")

    class_encod=df.iloc[:,-1]
    converter=LabelEncoder()
    y=converter.fit_transform(class_encod)

    df=df.drop(labels="filename",axis=1)

    fit=StandardScaler()
    X=fit.fit_transform(np.array(df.iloc[:,:-1],dtype=float))
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

    
    model = keras.models.load_model("good.h5")
    
    with open('model_history.pkl', 'rb') as file:
        loaded_model_history = pickle.load(file)
    
    
    test_loss,test_acc=model.evaluate(X_test,y_test,batch_size=256)
    print("The test loss is ",test_loss)
    print("The best accuracy is: ",test_acc*100)

    Validation_plot(loaded_model_history)


    sample = X_test
    sample = sample[np.newaxis, ...]
    prediction = model.predict(X_test)
    predicted_index = np.argmax(prediction, axis = 1)
    cm = confusion_matrix(y_test,predicted_index)
    print(cm)
    
    plt.plot(loaded_model_history['loss'])
    plt.plot(loaded_model_history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
    
if __name__ == "__main__":
    main()
