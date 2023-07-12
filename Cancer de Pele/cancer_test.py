#create new file test.py and run this file
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import tensorflow as tf
import os
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import img_to_array
from keras.initializers import glorot_uniform
from sklearn.preprocessing import LabelEncoder
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout 

import cv2

import numpy as np

img_size=224
pesos_path='cancer_pele.h5'

def build_model():
    model = tf.keras.Sequential()

    model.add(Conv2D(32,3,padding="same", activation="relu", input_shape=(224,224,3)))
    model.add(MaxPool2D())
    model.add(Conv2D(32, 3, padding="same", activation="relu"))
    model.add(MaxPool2D())
    model.add(Conv2D(64, 3, padding="same", activation="relu"))
    model.add(MaxPool2D())
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(128,activation="relu"))
    model.add(Dense(2, activation="softmax"))

    # you can either compile or not the model
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    return model

def showimg(img):
  cv2.imshow('image',img)
  # cv2.waitKey(0)
  cv2.destroyAllWindows()  

def read_img(img_path):
  img_path = img_path.replace(' ', '\\ ')
  if  not os.path.isfile(img_path): 
    print('arquivo nao existe ',img_path)
    raise AssertionError('Arquivo nao existe')

  print('Lendo o arquivo ',img_path )
  img = cv2.imread(img_path)[...,::-1]
  showimg(img)
  print('Convertendo cor do arquivo ',img_path )
  # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  # img = cv2.resize(img,(200,200))
  # img = img/255 # normalising
  return img

def get_model(model_path,pesos_path):
  model_path = model_path.replace(' ', '\\ ')
  if  not os.path.isfile(model_path): 
    print('Modelo nao existe ',model_path)
    raise AssertionError('Modelo nao existe')

  #Reading the model from JSON file
  with open(model_path, 'r') as json_file:
      json_savedModel= json_file.read()
  #load the model architecture 
  
  model = tf.keras.models.model_from_json(json_savedModel)  
  
  model.compile('adam', 'binary_crossentropy', metrics='accuracy')
  model1=model.load_weights(pesos_path)
  model.summary()
  return model1

# C:\\bc_niteroi_rj\\ML_SAUDE\\cancer_pele\\resnet50_cancer_model.h5
#load saved model
# model = tf.keras.models.load_model('resnet50_cancer_model.json') 





def show(img):
  cv2.imshow('image',img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()


def predict_img(img_path,modelo):
  test = []
  img=read_img(img_path)[...,::-1] #convert BGR to RGB format
  print(img.shape)
#  show(img)
  img=cv2.resize(img, (img_size, img_size))
  img = image.img_to_array(img)
  img = np.expand_dims(img, axis = 0)
#  print(img.shape)  
#  show(img)
  

  ''' 
  x = image.img_to_array(img)
  x = img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)  
  newarray = np.array(img) / 255
  newarray=np.zeros(1,img_size, img_size,3)
  '''

  preds =  modelo.predict(img).astype("int32")

#   preds=modelo.predict(newarray, verbose=1) 
  # Generate arg maxes for predictions
  print('PREDS ',preds)
  classes = np.argmax(preds, axis = 1)
  print(classes)
  # create a list containing the class labels
  class_labels = ['0','1'] # ,’Orange’
  # find the index of the class with maximum score
  pred = np.argmax(preds, axis=-1)
  # print the label of the class with maximum score
  print(class_labels[pred[0]])
  return img,pred


#model_path="resnet50_cancer_model.json"
model_path="classifica1_model.json"
model_img=get_model(model_path,pesos_path)
model1=build_model()
model1.load_weights(pesos_path)

test = []
test_y = []

#img_path = r"8.jpg"

#predict_img(img_path,model1)

# exit()
test_path='C://bc_niteroi_rj//ML_SAUDE//cancer_pele//train//benign//'
test_path='C://bc_niteroi_rj//ML_SAUDE//cancer_pele//Photos-001//'
ben = os.listdir(test_path)
f=0
for i in ben:
  f=f+1
  x=test_path+i
  print(x)
  img,pred=predict_img(x,model1) 
  test.append(img)
  test_y.append(pred)
#  if f>10:
#    break

exit()
test_path='C://bc_niteroi_rj//ML_SAUDE//cancer_pele//train//malignant//'
mal = os.listdir(test_path)
f=0
for i in mal:
  f=f+1
  x=test_path+i
  print(x)
  img,pred=predict_img(x,model1) 
  test.append(img)
  test_y.append(pred)
  if f>10:
    break
  

