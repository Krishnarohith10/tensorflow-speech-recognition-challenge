import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.api import keras

path = 'D:\\tensorflow_speech_recognition_challenge'
train_path = path+'\\train\\audio'
test_path = path+'\\test\\audio'

labels = os.listdir(train_path)

all_waves = []
all_labels=[]
for label in labels:
    print("Label:", label)
    waves = [f for f in os.listdir(train_path+'\\'+label) if f.endswith('.wav')]
    for wave in tqdm(waves):
        samples, sample_rate=librosa.load(train_path+'\\'+label+'\\'+wave, sr=16000)
        samples = librosa.resample(samples, sample_rate, 8000)
        if(len(samples)==8000):
            all_waves.append(samples)
            all_labels.append(label)

le = LabelEncoder()
y = le.fit_transform(all_labels)
y = keras.utils.to_categorical(y, num_classes=len(labels))

all_waves = np.array(all_waves).reshape(-1, 8000, 1)

x_train, x_val, y_train, y_val = train_test_split(all_waves, y, test_size=0.2, 
                                                  random_state=777, shuffle=True)

del all_waves, all_labels, y

inputs = keras.layers.Input(shape=(8000,1))

conv = keras.layers.BatchNormalization()(inputs)
conv = keras.layers.Dropout(0.1)(conv)
conv = keras.layers.Conv1D(16, kernel_size=13, strides=1)(conv)
conv = keras.layers.Activation('relu')(conv)

conv = keras.layers.BatchNormalization()(conv)
conv = keras.layers.Dropout(0.1)(conv)
conv = keras.layers.Conv1D(32, kernel_size=11, strides=1)(conv)
conv = keras.layers.Activation('relu')(conv)

conv = keras.layers.BatchNormalization()(conv)
conv = keras.layers.Dropout(0.1)(conv)
conv = keras.layers.Conv1D(64, kernel_size=9, strides=1)(conv)
conv = keras.layers.Activation('relu')(conv)

conv = keras.layers.BatchNormalization()(conv)
conv = keras.layers.Dropout(0.1)(conv)
conv = keras.layers.Conv1D(128, kernel_size=7, strides=1)(conv)
conv = keras.layers.Activation('relu')(conv)

maxp = keras.layers.MaxPooling1D(pool_size=2, strides=2)(conv)

#Flatten layer
conv = keras.layers.Flatten()(maxp)

#Dense Layer 1
conv = keras.layers.Dense(256, activation='relu')(conv)
conv = keras.layers.Dropout(0.3)(conv)

#Dense Layer 2
conv = keras.layers.Dense(128, activation='relu')(conv)
conv = keras.layers.Dropout(0.3)(conv)

outputs = keras.layers.Dense(len(labels), activation='softmax')(conv)

model = keras.models.Model(inputs, outputs)

model.summary()

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, 
                                   patience=10, min_delta=0.0001) 
mc = keras.callbacks.ModelCheckpoint('best_model.h5', monitor='val_acc', verbose=1, 
                                     save_best_only=True, mode='max')

history=model.fit(x_train, y_train, epochs=100, callbacks=[es,mc], batch_size=64, 
                  validation_data=(x_val,y_val))

plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.title('Accuracy vs Validation Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.title('Loss vs Validation Loss')
plt.legend()
plt.show()

