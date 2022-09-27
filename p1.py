import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import tensorflow as tf 
from tensorflow.keras import models,layers

####Read in pandas frame
dftrain_raw = pd.read_csv('train.csv')
dftest_raw = pd.read_csv('test.csv')
#['key']   ,  [['key1','key2',....]]   ,  [int:int] slicing row  ,   df.iloc[]   , df.loc[]    ,df.loc[int:int,:'key']
#dftrain_raw.head(10)

####Exploratory Data Analysis
ax = dftrain_raw['Survived'].value_counts().plot(kind = 'bar',
     figsize = (8,4),fontsize=15,rot = 0)
ax.set_ylabel('Counts',fontsize = 15)
ax.set_xlabel('Survived',fontsize = 15)
plt.show()


ax = dftrain_raw['Age'].plot(kind = 'hist',bins = 60,color= 'purple',
                        figsize = (12,8),fontsize=15)
ax.set_ylabel('Frequency',fontsize = 15)
ax.set_xlabel('Age',fontsize = 15)
plt.show()


#return of query is dataframe
ax = dftrain_raw.query('Survived == 0')['Age'].plot(kind = 'density',  
                      figsize = (12,8),fontsize=15)
dftrain_raw.query('Survived == 1')['Age'].plot(kind = 'density',
                      figsize = (12,8),fontsize=15)
ax.legend(['Survived==0','Survived==1'],fontsize = 12)
ax.set_ylabel('Density',fontsize = 15)
ax.set_xlabel('Age',fontsize = 15)
plt.show()



def preprocessing(dfdata):

    dfresult= pd.DataFrame()
    #Pclass
    dfPclass = pd.get_dummies(dfdata['Pclass'])
    dfPclass.columns = ['Pclass_' +str(x) for x in dfPclass.columns ]
    dfresult = pd.concat([dfresult,dfPclass],axis = 1)
    dfSex = pd.get_dummies(dfdata['Sex'])
    dfresult = pd.concat([dfresult,dfSex],axis = 1)
    #Age
    dfresult['Age'] = dfdata['Age'].fillna(0)
    dfresult['Age_null'] = pd.isna(dfdata['Age']).astype('int32')
    #SibSp,Parch,Fare
    dfresult['SibSp'] = dfdata['SibSp']
    dfresult['Parch'] = dfdata['Parch']
    dfresult['Fare'] = dfdata['Fare']
    #Carbin
    dfresult['Cabin_null'] =  pd.isna(dfdata['Cabin']).astype('int32')
    #Embarked
    dfEmbarked = pd.get_dummies(dfdata['Embarked'],dummy_na=True)
    dfEmbarked.columns = ['Embarked_' + str(x) for x in dfEmbarked.columns]
    dfresult = pd.concat([dfresult,dfEmbarked],axis = 1)
    return(dfresult)
    

x_train = preprocessing(dftrain_raw)
y_train = dftrain_raw['Survived'].values
x_test = preprocessing(dftest_raw)
y_test = dftest_raw['Survived'].values
####Build model Sequential. Others:API,Inheritance
tf.keras.backend.clear_session()

model = models.Sequential()
model.add(layers.Dense(20,activation = 'relu',input_shape=(15,)))
model.add(layers.Dense(10,activation = 'relu' ))
model.add(layers.Dense(1,activation = 'sigmoid' ))

model.summary()


###Train by fit

model.compile(optimizer='adam',
            loss='binary_crossentropy',
            metrics=['AUC'])

history = model.fit(x_train,y_train,
                    batch_size= 64,
                    epochs= 30,
                    validation_split=0.2 #for validation
                   )
def plot_metric(history, metric):
    train_metrics = history.history[metric]
    val_metrics = history.history['val_'+metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')
    plt.title('Training and validation '+ metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_"+metric, 'val_'+metric])
    plt.show()


plot_metric(history,"loss")

plot_metric(history,"AUC")


###Use model to predict
model.predict(x_test[0:10])

model.predict_classes(x_test[0:10])



###Save model structure and weights
model.save('keras_model_weight.h5')

###Load
model = models.load_model('keras_model_weight.h5')
model.evaluate(x_test,y_test)


###Save model structure
json_str = model.to_json()
###Load structure
model_json = models.model_from_json(json_str)
###Save weights
model.save_weights('weight.h5')
###Construct model from json
model_json = models.model_from_json(json_str)
model_json.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['AUC']
    )
###Load weight for model
model_json.load_weights('weight.h5')
model_json.evaluate(x_test,y_test)