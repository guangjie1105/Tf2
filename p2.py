"""Cifar2 with two categories plane and automobile 1000 for each 
In tensorflow normally has two method for data preparation 1.tf.keras ImageDataGenerator   2.tf.data.Dataset with tf.image
Pick first one in this script """
#train:5000 each
#test:1000 each
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets,layers,models
from matplotlib import pyplot as plt
####Data prepa

##Load data remote

cifar10 = tf.keras.datasets.cifar10

(x_train,y_train),(x_test,y_test) = cifar10.load_data() #tuple with length 2
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

##Data augmentation for train
train_datagen = ImageDataGenerator(
            rescale = 1./255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

##test no need augmentation
test_datagen = ImageDataGenerator(rescale=1./255)

##Load image from local
""" fnames = [os.path.join('/home/uif15692/Tensorflow2_project', fname) for 
          fname in os.listdir('/home/uif15692/Tensorflow2_project')]

img_path = fnames[3]
img = image.load_img(img_path, target_size=(32, 32))
x = image.img_to_array(img)
plt.figure(1,figsize = (10,8))
plt.subplot(2,2,1)
plt.imshow(image.array_to_img(x))
plt.title('original image') """




#x shape (32,32,3)
""" x = x.reshape((1,) + x.shape)
np.expend_dims(x,axis=0) """


###Image show after data aug
""" i = 0
for batch in train_datagen.flow(x_train, batch_size=1):
    #batch (1,32,32,3)
    plt.subplot(2,2,i+1)
    plt.imshow(keras.utils.array_to_img(batch[0]))
    plt.title('after augumentation %d'%(i+1))
    i = i +  1
    if i % 4 == 0:
        break
plt.show()
train_datagen.fit(x_test)
train_datagen.fit(x_test) """
train_generator = test_datagen.flow(x_train ,y_train, batch_size=32)
test_generator =  test_datagen.flow(x_test,y_test,batch_size=32)


#### Build model by API

tf.keras.backend.clear_session() 
""" inputs = layers.Input(shape=(32,32,3))
x = layers.Conv2D(32,kernel_size=(3,3))(inputs)
x = layers.MaxPool2D()(x)
x = layers.Conv2D(64,kernel_size=(5,5))(x)
x = layers.MaxPool2D()(x)
x = layers.Dropout(rate=0.1)(x)
x = layers.Flatten()(x)
x = layers.Dense(80,activation='relu')(x)
outputs = layers.Dense(10,activation = 'sigmoid')(x)

model = models.Model(inputs = inputs,outputs = outputs) """
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')])


model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'])
import numpy as np 
train_steps_per_epoch  = np.ceil(50000/32)
test_steps_per_epoch  = np.ceil(10000/32)

history = model.fit(
        train_generator,
        steps_per_epoch = train_steps_per_epoch,
        epochs = 10,batch_size=32,
        validation_data= test_generator,
        validation_steps=test_steps_per_epoch,
        use_multiprocessing=False #linux上可使用多进程读取数据
        )
#x_train, x_test = x_train / 255.0, x_test / 255.0
""" model.fit(train_datagen.flow(x_train, y_train, batch_size=32,
         subset='training'),
         validation_data=train_datagen.flow(x_test, y_test,
         batch_size=8, subset='validation'),
         steps_per_epoch=len(x_train) / 32, epochs=10) """

#model.fit(x_train, y_train, epochs=15, validation_data=(x_test, y_test))
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

train_generator = train_datagen.flow(x_train,y_train ,batch_size=32,)