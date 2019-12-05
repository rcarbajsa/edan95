from keras import layers, optimizers, models
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.applications import InceptionV3

def data_augmentation_convnet():
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu',
                                input_shape=(150, 150, 3)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(5, activation='sigmoid'))

        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.RMSprop(lr=1e-4),
                      metrics=['acc'])
        train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,)

        # Note that the validation data should not be augmented!
        test_datagen = ImageDataGenerator(rescale=1./255)
        train_dir = '/home/rcarbajsa/edan95/datasets/flowers_split/train'
        validation_dir = '/home/rcarbajsa/edan95/datasets/flowers_split/validation'
        test_dir = '/home/rcarbajsa/edan95/datasets/flowers_split/test' 
        train_generator = train_datagen.flow_from_directory(
                # This is the target directory
                train_dir,
                # All images will be resized to 150x150
                target_size=(150, 150),
                batch_size=32,
                # Since we use binary_crossentropy loss, we need binary labels
                class_mode='categorical')

        validation_generator = test_datagen.flow_from_directory(
                validation_dir,
                target_size=(150, 150),
                batch_size=24,
                class_mode='categorical')

        history = model.fit_generator(
              train_generator,
              steps_per_epoch=173,
              epochs=100,
              validation_data=validation_generator,
              validation_steps=50)
        model.save('flowers2.h5')
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
        test_generator = test_datagen.flow_from_directory(
                test_dir,
                target_size=(150, 150),
                batch_size=20,
                class_mode='categorical')       
        test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)
        print('test acc:', test_acc)

def normal_convnet():
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu',
                                input_shape=(150, 150, 3)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(5, activation='sigmoid'))
        model.summary() 
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.RMSprop(lr=1e-4),
                      metrics=['acc'])  
        # All images will be rescaled by 1./255
        train_datagen = ImageDataGenerator(rescale=1./255)
        test_datagen = ImageDataGenerator(rescale=1./255)       
        train_dir = '/home/rcarbajsa/edan95/datasets/flowers_split/train'
        validation_dir = '/home/rcarbajsa/edan95/datasets/flowers_split/validation'
        test_dir = '/home/rcarbajsa/edan95/datasets/flowers_split/test' 
        train_generator = train_datagen.flow_from_directory(
                # This is the target directory
                train_dir,
                # All images will be resized to 150x150
                target_size=(150, 150),
                batch_size=15,
                # Since we use binary_crossentropy loss, we need binary labels
                class_mode='categorical')       
        validation_generator = test_datagen.flow_from_directory(
                validation_dir,
                target_size=(150, 150),
                batch_size=5,
                class_mode='categorical') 
        history = model.fit_generator(
              train_generator,
              steps_per_epoch=173,
              epochs=30,
              validation_data=validation_generator,
              validation_steps=173)     
        model.save('flowers1.h5')       
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
        test_generator = test_datagen.flow_from_directory(
                test_dir,
                target_size=(150, 150),
                batch_size=20,
                class_mode='categorical')       
        test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)
        print('test acc:', test_acc)

def pretrained_convnet():
        base_model = InceptionV3(weights='imagenet', include_top=False)
        train_dir = '/home/rcarbajsa/edan95/datasets/flowers_split/train'
        validation_dir = '/home/rcarbajsa/edan95/datasets/flowers_split/validation'
        test_dir = '/home/rcarbajsa/edan95/datasets/flowers_split/test' 
        datagen = ImageDataGenerator(rescale=1./255)
        batch_size = 20

        def extract_features(directory, sample_count):
                features = np.zeros(shape=(sample_count, 4, 4, 512))
                labels = np.zeros(shape=(sample_count))
                generator = datagen.flow_from_directory(
                        directory,
                        target_size=(150, 150),
                        batch_size=batch_size,
                        class_mode='binary')
        i = 0
        for inputs_batch, labels_batch in generator:
                features_batch = conv_base.predict(inputs_batch)
                features[i * batch_size : (i + 1) * batch_size] = features_batch
                labels[i * batch_size : (i + 1) * batch_size] = labels_batch
                i += 1
                if i * batch_size >= sample_count:
                        # Note that since generators yield data indefinitely in a loop,
                        # we must `break` after every image has been seen once.
                        break
        return features, labels

        train_features, train_labels = extract_features(train_dir, 2000)
        validation_features, validation_labels = extract_features(validation_dir, 1000)
        test_features, test_labels = extract_features(test_dir, 1000)

def main():
        data_augmentation_convnet()
            
if __name__ == "__main__":
    main()