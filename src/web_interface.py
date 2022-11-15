import streamlit as st
import plotly.express as px
import pandas as pd
from streamlit import config
import numpy as np
import cv2


import keras
import tensorflow as tf
from keras.backend import batch_dot
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array, load_img
from keras.utils import np_utils
from keras import callbacks
from keras.engine.input_layer import Input
from keras.models import Model, Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, ReLU, Flatten, Dense, GlobalAveragePooling2D, BatchNormalization, Dropout, Lambda
import tensorflow.keras.backend as backend

IMG_SHAPE = (238,320,1)
IMG_WH = (238,320)


def build_siamese_model(img_shape):
    """
    Genera la arquitectura de la red gemela
    """
    model = Sequential()
    model.add(Conv2D(96, (11,11), activation='relu', input_shape=img_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
    model.add(Conv2D(256, (5,5), strides=(1,1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
    model.add(Conv2D(384, (3,3), strides=(1,1), padding='same', activation='relu'))
    model.add(Conv2D(256, (3,3), strides=(1,1), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(5,5)))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(128))

    return model

def euclidean_distance(vectors):
    """
    Calcula la distancia euclidea entre dos vectores
    """
    x, y = vectors
    sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
    return tf.math.sqrt(tf.math.maximum(sum_square, backend.epsilon()))

    return distance

def loadImg(path):
    image = img_to_array(load_img(path, color_mode="grayscale", target_size=IMG_WH, interpolation="bilinear")).astype(np.float32)
    #image = np.expand_dims(image, axis=0)
    return image

def apply_clahe(image,crop):
    clahe = cv2.createCLAHE(clipLimit=crop, tileGridSize=(8, 8))
    image= clahe.apply(image.astype(np.uint8)).astype(np.float32)
    return image

def preprocess(img):
	# cropped = img[30:210,30:300]
	# cropped = apply_clahe(cropped, 4.0)
	# return cropped

	# Si queremos eliminar el preprocesado:
	return img

def read_img_file(img):
	return cv2.imdecode(np.asarray(bytearray(img.read())), 0)



# Definición de los parámetros de entrenamiento.
img1 = Input(shape=IMG_SHAPE)
img2 = Input(shape=IMG_SHAPE)

feature_extractor = build_siamese_model(IMG_SHAPE)
# feature_extractor.summary()

features1 = feature_extractor(img1)
features2 = feature_extractor(img2)

distance = Lambda(euclidean_distance)([features1,features2])
outputs = Dense(1, activation="sigmoid") (distance)

model = Model(inputs=[img1, img2], outputs=outputs)
model.load_weights('../bin/modelo-iris.h5')

db_images_paths = ['../dataset/MMU-Iris-2/020205.bmp', '../dataset/MMU-Iris-2/500204.bmp', '../dataset/MMU-Iris-2/200102.bmp', 
         '../dataset/MMU-Iris-2/330104.bmp', '../dataset/MMU-Iris-2/1000201.bmp', '../dataset/MMU-Iris-2/990102.bmp', '../dataset/MMU-Iris-2/280102.bmp',
         '../dataset/MMU-Iris-2/110101.bmp', '../dataset/MMU-Iris-2/160104.bmp', '../dataset/MMU-Iris-2/880101.bmp', '../dataset/MMU-Iris-2/330203.bmp']

db_images = []
for i in db_images_paths:
  db_images.append(loadImg(i))


st.title("Iris recognition system")
st.subheader("Upload a picture of your eye!")
img=st.file_uploader("")
coincidence=[]
if img is not None:
	st.image(img)
	# img_array = np.asarray(bytearray(img.read()))
	np_img = preprocess( read_img_file(img) )
	# np_img = cv2.resize(np_img, dsize=IMG_WH, interpolation=cv2.INTER_CUBIC)
	np_img = np.expand_dims(np_img,axis=0)
	for db_img in db_images:
		db_img_preproc = np.expand_dims(preprocess(db_img),axis=0)
		coincidence.append(model.predict( [np_img,db_img_preproc], steps=1 )[0,0])

#Si la probabilidad de que coincida con una imagen de nuestra bases de datos es baja en todas
if len(coincidence)>0:
	ordered_indices = np.argsort(coincidence)[::-1]
	print(ordered_indices)

	if np.max(coincidence) < 0.5:
		st.subheader("We are sorry, but this person is not registered in our database")

	else:
		#Si coincide con una imagen con alta probabilidad
		st.subheader("This person seems registered in our database.")
	
	st.subheader("The images in our database with the highest coincidence probabilities are the following:")

	for i in ordered_indices[0:5]:
		col1, col2, col3, col4 = st.columns(4)
		coincidence_perc = int(coincidence[i]*100)
		with col1:
			#st.text("image")
			st.image(db_images[i]/255.0)
		with col2:
			my_bar = st.progress(coincidence_perc)
			#st.write(my_bar)
		with col3:
			st.text(f"{coincidence_perc}%")
		with col4:
			st.text(f"ID {db_images_paths[i].split('/')[-1].split('.')[0][0:-2]}")
			
# st.subheader("This is the corresponding image that best matches your eye:")

