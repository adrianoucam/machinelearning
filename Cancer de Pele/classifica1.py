# https://www.analyticsvidhya.com/blog/2020/10/create-image-classification-model-python-keras/
# https://hsf-training.github.io/hsf-training-ml-gpu-webpage/aio/index.html

# https://www.tensorflow.org/install/source_windows?hl=pt-br#gpu
'''

Observação: a partir do TF 2.11, a compilação CUDA não é compatível com o Windows. Para usar a GPU do TensorFlow no Windows, você precisará criar/instalar o TensorFlow no WSL2 ou usar o tensorflow-cpu com o TensorFlow-DirectML-Plugin
https://towardsdatascience.com/how-to-finally-install-tensorflow-gpu-on-windows-10-63527910f255

'''

import matplotlib.pyplot as plt
import seaborn as sns

import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout 
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.metrics import precision_score , recall_score

from sklearn.metrics import classification_report,confusion_matrix
from keras.callbacks import History 

from sklearn.metrics import roc_curve, auc


history = History()

import tensorflow as tf

import cv2
import os

import numpy as np

def plot_roc_curve(fpr,tpr): 
  plt.plot(fpr,tpr) 
  plt.axis([0,1,0,1]) 
  plt.xlabel('False Positive Rate') 
  plt.ylabel('True Positive Rate') 
  plt.show()    
   
class ModelMetrics(tf.keras.callbacks.Callback):
   
  def on_train_begin(self,logs={}):
    self.precisions=[]
    self.recalls=[]
    self.f1_scores=[]
  def on_epoch_end(self, batch, logs={}):
     
    y_val_pred=self.model.predict_classes(x_val)
    
    _precision,_recall,_f1,_sample=score(y_val,y_val_pred)  
     
    self.precisions.append(_precision)
    self.recalls.append(_recall)
    self.f1_scores.append(_f1)

labels = ['benign', 'malignant']

Run_epochs = 120
img_size = 224

def get_data(data_dir):
    data = [] 
    for label in labels: 
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img))[...,::-1] #convert BGR to RGB format
                resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return np.array(data)

# C:\bc_niteroi_rj\ML_SAUDE\cancer_pele\train\malignant

train = get_data('C:\\bc_niteroi_rj\\ML_SAUDE\\cancer_pele\\train\\')

# train = get_data('C:\\bc_niteroi_rj\\ML_SAUDE\\cancer_pele\\train\\')

val = get_data('C:\\bc_niteroi_rj\\ML_SAUDE\\cancer_pele\\test\\')

l = []
for i in train:
    if(i[1] == 0):
        l.append("benign")
    else:
        l.append("malignant")

sns.set_style('darkgrid')
#sns.countplot(l)

plt.figure(figsize = (5,5))
plt.imshow(train[1][0])
plt.title(labels[train[0][1]])
plt.show()

plt.figure(figsize = (5,5))
plt.imshow(train[-1][0])
plt.title(labels[train[-1][1]])
plt.show()

x_train = []
y_train = []
x_val = []
y_val = []

for feature, label in train:
  x_train.append(feature)
  y_train.append(label)

for feature, label in val:
  x_val.append(feature)
  y_val.append(label)

# Normalize the data
x_train = np.array(x_train) / 255
x_val = np.array(x_val) / 255

x_train.reshape(-1, img_size, img_size, 1)
y_train = np.array(y_train)

x_val.reshape(-1, img_size, img_size, 1)
y_val = np.array(y_val)

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range = 30,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.2, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip = True,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(x_train)


model = Sequential()
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

model.summary()

model.save('cancer_model2')

metrics=ModelMetrics()

opt = Adam(learning_rate=0.000001)
model.compile(optimizer = opt , loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) , metrics = ['accuracy'])

history = model.fit(x_train,y_train,epochs = Run_epochs , validation_data = (x_val, y_val)) 

# ,callbacks=['metrics'] não funcionou

model.save_weights('cancer_pele.h5')

print(history.history.keys())

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(Run_epochs)

plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# Obter as previsões do modelo para o conjunto de teste
y_pred_proba = model.predict(x_val)[:, 1]  # Probabilidades da classe positiva (malignant)

predict_x=model.predict(x_val)
y_val_pred=np.argmax(predict_x,axis=1)

fpr , tpr , thresholds = roc_curve ( y_val, y_val_pred )

plot_roc_curve (fpr,tpr)  

auc_score=roc_auc_score(y_val,y_val_pred)


print("AUC:", auc_score)
 

print('Recall')
print(recall_score(y_val,y_val_pred,average=None)) 
print('Precision Score')
print(precision_score(y_val,y_val_pred,average=None))
 
print(classification_report(y_val, y_val_pred))


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(Run_epochs)

plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# predictions =  (model.predict(x_val) > 0.5).astype("int32")

# era assim predictions = model.predict_classes(x_val)

# predictions = predictions.reshape(1,-1)[0]

# print(classification_report(y_val, predictions))
# ,target_names = ['Benign (Class 0)','malignant (Class 1)']
model_json = model.to_json()
with open("C://bc_niteroi_rj//ML_SAUDE//cancer_pele//classifica1_model.json", "w") as json_file:
  json_file.write(model_json)
