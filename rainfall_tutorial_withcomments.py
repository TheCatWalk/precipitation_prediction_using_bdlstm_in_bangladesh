import numpy as np
from pandas.plotting import register_matplotlib_converters
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
from pylab import rcParams
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from keras.layers import (
    Input,
    Dense,
    LSTM,
    AveragePooling1D,
    TimeDistributed,
    Flatten,
    Bidirectional,
    Dropout
)
from sklearn import metrics
from keras.models import Model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from keras.callbacks import CSVLogger

#mape
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


tf.keras.backend.clear_session()
register_matplotlib_converters()
sns.set(style='whitegrid', palette='muted', font_scale=1.5)

rcParams['figure.figsize'] = 22, 10

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

#reading from CSV
df = pd.read_csv("D:\RUET\Thesis Papers\Agri\Data\yosef\customized_daily_rainfall_data_Copy.csv")
#droping bad data
df = df[df.Rainfall != -999]

#droping dates (leapyear, wrong day numbers of month)
df.drop(df[(df['Day']>28) & (df['Month']==2) & (df['Year']%4!=0)].index,inplace=True)
df.drop(df[(df['Day']>29) & (df['Month']==2) & (df['Year']%4==0)].index,inplace=True)
df.drop(df[(df['Day']>30) & ((df['Month']==4)|(df['Month']==6)|(df['Month']==9)|(df['Month']==11))].index,inplace=True)

#date parcing (Index)
date = [str(y)+'-'+str(m)+'-'+str(d) for y, m, d in zip(df.Year, df.Month, df.Day)]
df.index = pd.to_datetime(date)

df['Date'] = df.index
df['Dayofyear']=df['Date'].dt.dayofyear
df.drop('Date',axis=1,inplace=True)
df.drop(['Station'],axis=1,inplace=True)
df.head()


#limiting the dataframe to just rows where StationIndex is X
datarange = df.loc[df['StationIndex'] == 1]

#splitting train and test set
train_size = int(len(datarange) * 0.9)
test_size = len(datarange) - train_size
train, test = df.iloc[0:train_size], df.iloc[train_size:len(datarange)]

#Scaling the feature and label columns of the dataset
from sklearn.preprocessing import RobustScaler
f_columns = ['Year', 'Month','Day','Dayofyear']
f_transformer = RobustScaler()
l_transformer = RobustScaler()
f_transformer = f_transformer.fit(train[f_columns].to_numpy())
l_transformer = l_transformer.fit(train[['Rainfall']])


train.loc[:, f_columns] = f_transformer.transform(train[f_columns].to_numpy())
train['Rainfall'] = l_transformer.transform(train[['Rainfall']])
test.loc[:, f_columns] = f_transformer.transform(test[f_columns].to_numpy())
test['Rainfall'] = l_transformer.transform(test[['Rainfall']])

#making smaller train and test sections withing the dataset
def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].to_numpy()
        Xs.append(v)        
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 10

# define and prepare the dataset
def get_data():
    # reshape to [samples, time_steps, n_features]
    X_train, y_train = create_dataset(train, train.Rainfall, time_steps)
    X_test, y_test = create_dataset(test, test.Rainfall, time_steps)
    return X_train, y_train, X_test, y_test

#testing
#X_test[0][0]

# define and fit the model
def get_model(X_train, y_train):
    #model code
    model = keras.Sequential()   
    #3 biderectional LSTM layers
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(units=128, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences = True)))
    model.add(keras.layers.Dropout(rate=0.1))
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(units=128,  return_sequences = True)))
    model.add(keras.layers.Dropout(rate=0.1))
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(units=128 )))
    model.add(keras.layers.Dropout(rate=0.1))
    model.add(keras.layers.Dense(units=1, activation="relu"))
    model.compile(loss="mse", optimizer="Adam",metrics=['acc'])    
    #training the model
    #csv_logger = CSVLogger('D:\RUET\ModelSave\ModelHistory\historyStation17.csv', append=True, separator=',')
    history = model.fit(
        X_train, y_train, 
        epochs=500, 
        batch_size=1200, 
        validation_split=0.2,
        shuffle=False,
        #callbacks=[csv_logger]
    )
    return model


#get data and model
X_train, y_train, X_test, y_test = get_data()
model = get_model(X_train, y_train)
 
#saving the model
model.save("D:\RUET\ModelSave\ModelHistory\station17.h5")



#load a model - not needed for new epoch run
#from keras.models import load_model
keras.models.load_model('D:\RUET\ModelSave\ModelHistory\station1.h5', compile = True)
loaded_model.summary()

#Using text dataset to do a prediction
y_pred = model.predict(X_test, verbose=0)
y_pred_classes = model.predict_classes(X_test, verbose=0)


#inverst transformation
y_train_inv = l_transformer.inverse_transform(y_train.reshape(1, -1))
y_test_inv = l_transformer.inverse_transform(y_test.reshape(1, -1))
y_pred_inv = l_transformer.inverse_transform(y_pred)
y_pred_classes_inv = l_transformer.inverse_transform(y_pred_classes)

# reduce to 1d array
y_pred_inv = y_pred_inv[:, 0]
y_pred_classes_inv = y_pred_classes_inv[:, 0]
y_test_inv = y_test_inv[0, :]
#scoring and metrics section


# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_test_inv, y_pred_classes_inv)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(y_test_inv, y_pred_classes_inv, average='weighted', labels=np.unique(y_pred_classes_inv))
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(y_test_inv, y_pred_classes_inv, average='weighted')
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_test_inv, y_pred_classes_inv, average='weighted')
print('F1 score: %f' % f1)
#score
mse_score = np.sqrt(metrics.mean_squared_error(y_pred_inv,y_test_inv))
print('MSE: %f' %mse_score)
mae_score = np.sqrt(metrics.mean_absolute_error(y_pred_inv,y_test_inv))
print('MAE: %f' %mae_score)



report = classification_report(y_test_inv, y_pred_classes_inv,labels=np.unique(y_pred_classes_inv), output_dict=True)
dfscore = pd.DataFrame(report).transpose()
dfscore.to_csv('D:\RUET\ModelSave\ModelHistory\historyStation17_classification_report.csv', sep = ',')

#plot and figure section



#plot true vs pred together
plt.plot(y_test_inv.flatten(), 'b', marker='o', label="true")
plt.plot(y_pred_inv.flatten(), 'r', label="prediction")
plt.ylabel('Rainfall')
plt.xlabel('Time Step')
plt.legend()
plt.show();



#plot true vs pred separately
plt.subplot(221)
plt.plot(y_test_inv.flatten(), 'b', label="true")
plt.legend()
plt.subplot(222)
plt.plot(y_pred_inv.flatten(), 'r', label="prediction")
plt.ylabel('Rainfall')
plt.xlabel('Time Step')
plt.legend()
plt.show();




#plot full timeline
plt.plot(np.arange(0, len(y_train)), y_train_inv.flatten(), 'g', label="history")
plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_test_inv.flatten(), marker='.', label="true")
plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_pred_inv.flatten(), 'r', label="prediction")
plt.ylabel('Rainfall')
plt.xlabel('Time Step')
plt.legend()
plt.show();

#plot loss total
loss = model.history.history['loss']
epochs=range(len(loss))
plt.plot(epochs, loss, 'r', linewidth=5)
plt.title('Training loss', fontsize=40)
plt.xlabel("Epochs", fontsize=40)
plt.ylabel("Loss", fontsize=40)
plt.legend(["Loss"], fontsize=40)
plt.figure()

#plot loss zoomed
zoomed_loss = loss[300:]
zoomed_epochs = range(300,500)
plt.plot(zoomed_epochs, zoomed_loss, 'r', linewidth=5)
plt.title('Training loss', fontsize=40)
plt.xlabel("Epochs", fontsize=40)
plt.ylabel("Loss", fontsize=40)
plt.legend(["Loss"], fontsize=40)
plt.figure()

#plot loss vs validation
val_loss = model.history.history['val_loss']
epochs=range(len(val_loss))
plt.plot(epochs, val_loss, 'r')
plt.plot(epochs, loss, 'b')
plt.xlabel("Epochs")
plt.ylabel("Loss vs val_loss")
plt.legend()
plt.figure()


#plot loss vs validation
zoomed_val_loss = val_loss[300:]
zoomed_epochs = range(300,500)
plt.plot(zoomed_epochs, zoomed_val_loss, 'r')
plt.plot(zoomed_epochs, zoomed_loss, 'b')
plt.xlabel("Epochs")
plt.ylabel("Loss vs val_loss")
plt.legend()
plt.figure()

