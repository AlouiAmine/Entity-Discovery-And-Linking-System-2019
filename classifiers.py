from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from keras import models
from keras import layers
from keras import optimizers
from keras.regularizers import l2
import keras
import argparse
import numpy as np
import json


parser = argparse.ArgumentParser()
parser.add_argument('--data',type=str, help='the training data', required=True)
parser.add_argument('--iter', type=int,help='number of iteration', required=True)
parser.add_argument('--batch_size',type=int, help='Batch size to train the deeplearning model', required=True)
parser.add_argument('--verbose',type=int, help='display the training progress', default=0)
args=parser.parse_args()

with open(args.data) as f:
    data = json.load(f)
    
data_x=[]
data_y=[]
for key in data:
    for embed in data[key]:
        data_x.append(embed)
        data_y.append(key)

tag=list(data.keys())
tag2idx = {t: i for i, t in enumerate(tag)}
idx2tag = {i: t for i, t in enumerate(tag)}
y_t=np.array([tag2idx[i] for i in data_y])



X_train, X_test, y_train, y_test = train_test_split(data_x,data_y,test_size=0.2, random_state=24)
X_test1, X_val, y_test1, y_val = train_test_split(X_test,y_test,test_size=0.5, random_state=24,stratify=y_test)



X_train_ = np.array(X_train).reshape(-1,776,1)
X_test_ = np.array(X_test).reshape(-1,776,1)
X_val_ = np.array(X_val).reshape(-1,776,1)

y_train_=np.array([tag2idx[i] for i in y_train])
y_test_=np.array([tag2idx[i] for i in y_test])
y_val_=np.array([tag2idx[i] for i in y_val])

y_train_ = keras.utils.to_categorical(y_train_, len(set(y_train)))
y_test_ = keras.utils.to_categorical(y_test_, len(set(y_train)))
y_val_ = keras.utils.to_categorical(y_val_, len(set(y_train)))
input_shape = (776,1)


# Define model

model = models.Sequential()
model.add(layers.Flatten(input_shape=input_shape))
model.add(layers.Dense(256, activation='relu', input_dim=input_shape, kernel_regularizer=l2(0.01)))
# model.add(layers.Dropout(0.))
model.add(layers.Dense(len(data.keys()), activation='softmax'))
#model.summary()

model.compile(loss=keras.losses.categorical_crossentropy,
             optimizer=optimizers.Adam(),
             metrics=['accuracy'])    
    
# Train model
history = model.fit(X_train_, y_train_, validation_data=[X_val_, y_val_],
             batch_size=args.batch_size, 
          epochs=args.iter,
          verbose=args.verbose)

print('evaluation on test data...')

predictions = model.predict(X_test_)
print('predictions shape:', predictions.shape)
from sklearn.metrics import classification_report
y_pred=[np.argmax(predictions[i]) for i in range(len(X_test_))]

y_test_t=y_test#[idx2tag[i] for i in y_test]
y_pred_=[idx2tag[i] for i in y_pred]
from sklearn.metrics import f1_score
print('micro f1-score\n')
print(f1_score(y_test_t, y_pred_, average='micro'))
print('macro f1-score\n')
print(f1_score(y_test_t, y_pred_, average='macro'))

print(classification_report(y_test_t,y_pred_ ))
model.save('classifier_deep_learning.h5')  # creates a HDF5 file 'my_model.h5'

