import os
import random
import shutil
import numpy as np
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import InceptionV3
import matplotlib.pyplot as plt




conv_base = InceptionV3(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))
conv_base.summary()


base_dir = 'C:/Users/gbaro/Documents/UNIVERSIDAD/Erasmus/edan95/edan95/labs/flowers_split/' 

train_dir = os.path.join(base_dir, 'train').replace("\\","/")
validation_dir = os.path.join(base_dir, 'validation').replace("\\","/")
test_dir = os.path.join(base_dir, 'test').replace("\\","/")

datagen = ImageDataGenerator(rescale=1./255)
batch_size = 5

def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 3, 3, 2048))
    labels = np.zeros(shape=(sample_count,5))
    generator = datagen.flow_from_directory(
            directory,
            target_size=(150, 150),
            batch_size=batch_size,
            class_mode='categorical')
    i = 0
    for inputs_batch, labels_batch in generator:
            #print(batch_size)
            print(i)
            print(labels_batch.size)
            print(inputs_batch.size)
            print(labels_batch.shape)
            features_batch = conv_base.predict(inputs_batch) #np.zeros(shape=(batch_size, 3, 3, 2048))
            print(features_batch.shape)
            print(features[i * batch_size : (i + 1) * batch_size].shape)
            features[i * batch_size : (i + 1) * batch_size] = features_batch
            labels[i * batch_size : (i + 1) * batch_size] = labels_batch
            i += 1
            if i * batch_size >= sample_count:
            # Note that since generators yield data indefinitely in a loop,
            # we must `break` after every image has been seen once.
                break
    return features,labels

train_features, train_labels = extract_features(train_dir, 2000)
validation_features, validation_labels = extract_features(validation_dir, 1000)
test_features, test_labels = extract_features(test_dir, 1000)

train_features=np.reshape(train_features,(2000,3*3*2048))
validation_features=np.reshape(validation_features,(1000,3*3*2048))
test_features=np.reshape(test_features,(1000,3*3*2048))

model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=3*3*2048))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(5, activation='softmax'))

model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='categorical_crossentropy',
              metrics=['acc'])

history = model.fit(train_features, train_labels,
                    epochs=30,
                    batch_size=32,
                    validation_data=(validation_features, validation_labels))

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()