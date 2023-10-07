import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data=pd.read_csv("winequality-red.csv")
data.head()

reviews=[]
for i in data['quality']:
    if i >= 1 and i <= 4:
        reviews.append(0)
    elif i >= 5 and i <= 6:
        reviews.append(1)
    elif i >= 7 and i <= 10:
        reviews.append(2)
data['reviews'] = reviews

x=data.drop("quality",axis=1)
x=x.drop("reviews",axis=1)
y=data["reviews"]

ss=StandardScaler()
x=ss.fit_transform(x)
x_train,x_test,y_train,y_test=train_test_split(x, y, test_size = 0.2, random_state = 42)

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
rbf_svc = SVC(C=1.5,gamma=1,kernel="rbf")
rbf_svc.fit(x_train, y_train)
rbf_svc_pred=rbf_svc.predict(x_test)
rbf_svc_conf_matrix = confusion_matrix(y_test, rbf_svc_pred)
rbf_svc_acc_score = accuracy_score(y_test, rbf_svc_pred)
print(rbf_svc_conf_matrix)
print(rbf_svc_acc_score*100)
a=[i/np.sum(i) for i in rbf_svc_conf_matrix]
lab=['Poor','medium','Good']
sns.heatmap(a,xticklabels=lab,yticklabels=lab,annot=True)

"""NN"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.colors import n_colors
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler,LabelEncoder
from sklearn.datasets import load_wine
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def NNtrain():
  my_model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(156,activation='linear'),
    tf.keras.layers.Dropout(0.30),
    tf.keras.layers.Dense(116,activation='LeakyReLU'),
    tf.keras.layers.Dense(200,activation='relu'),

    tf.keras.layers.Dense(190,activation='LeakyReLU'),

    tf.keras.layers.Dropout(0.40),
    tf.keras.layers.Dense(156,activation='softmax')
  ])
  optimiser = tf.keras.optimizers.Adam()
  my_model.compile (optimizer= optimiser, loss='sparse_categorical_crossentropy', metrics = ['accuracy'])
  history = my_model.fit(x_train, y_train, epochs=100, batch_size=40)
  return my_model


def NNPredict(my_model,user_test_x):
  my_predict = my_model.predict(user_test_x)
  return np.argmax(my_predict,axis=1)

model_mlp=NNtrain()


res=NNPredict(model_mlp,x_test)
print(np.array(y_test))
print(res)
dense_conf_matrix = confusion_matrix(y_test, res)
dense_acc_score = accuracy_score(y_test, res)
print(dense_conf_matrix)
print(dense_acc_score*100)

a=[i/np.sum(i) for i in dense_conf_matrix]
lab=['Poor','medium','Good']
sns.heatmap(a,xticklabels=lab,yticklabels=lab,annot=True)

"""卷积神经网络"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import matplotlib.image as mpimg
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier


import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

from keras import models,layers

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
from sklearn import preprocessing

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf


y_train_cnn = tf.one_hot(y_train,3)
x_train_cnn = np.delete(x_train,5,1)

from sklearn import preprocessing
X_train_cnn = x_train_cnn.reshape(-1, 2, 5, 1)
print('New shape of training data:', X_train_cnn.shape)

from sklearn import preprocessing
x_test_cnn = np.delete(x_test,5,1)
X_test_cnn = x_test_cnn.reshape(-1, 2, 5, 1)
print('New shape of training data:', X_test_cnn.shape)

y_test_cnn = tf.one_hot(y_test,3)

model = Sequential([
    Input(shape = (2, 5, 1)),
    Conv2D(32, 3, padding = 'same', activation = 'relu'),
    Conv2D(32, 3, padding = 'same', activation = 'relu'),
    Conv2D(32, 3, padding = 'same', activation = 'relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding = 'same', activation = 'relu'),
    Conv2D(64, 3, padding = 'same', activation = 'relu'),
    Conv2D(64, 3, padding = 'same', activation = 'relu'),
    Flatten(),
    Dropout(0.2),
    Dense(256, input_shape = (2, 5, 1), activation = 'relu'),
    Dense(128, activation = 'relu'),
    Dense(32, activation = 'relu'),
    Dense(3, activation = 'sigmoid')
])

model.summary()

model.compile(loss = 'binary_crossentropy',
              optimizer = Adam(),
              metrics = ['acc'])

reduce_lr = ReduceLROnPlateau(monitor = 'acc', patience = 3, verbose = 1, factor = 0.5, min_lr = 0.00001)

history = model.fit(X_train_cnn, y_train_cnn, epochs = 100)

def model_performance_graphs(classifier):

    fig, axes = plt.subplots(1, 2, figsize = (15, 8))

    axes[0].plot(classifier.epoch, classifier.history['acc'], label = 'acc')
    axes[0].set_title('Accuracy vs Epochs', fontsize = 20)
    axes[0].set_xlabel('Epochs', fontsize = 15)
    axes[0].set_ylabel('Accuracy', fontsize = 15)
    axes[0].legend()

    axes[1].plot(classifier.epoch, classifier.history['loss'], label = 'loss')
    axes[1].set_title("Loss Curve",fontsize=18)
    axes[1].set_xlabel("Epochs",fontsize=15)
    axes[1].set_ylabel("Loss",fontsize=15)
    axes[1].legend()

    plt.show()

model_performance_graphs(history)

cnn_train_acc = model.evaluate(X_train_cnn, y_train_cnn)[-1]
cnn_test_acc = model.evaluate(X_test_cnn, y_test_cnn)[-1]
print(cnn_train_acc, cnn_test_acc)

def result_CNN(X,model):
    x = X.tolist()
    Ey = []
    for i in range(len(x)):
      x[i].pop(5)
      temp = np.array(x[i]).reshape(-1,2,5,1)
      Ey.append(np.argmax(model(temp,training=False)))
    return Ey

sample_test_X = np.array([ 7.8   ,  0.88  ,  0.    ,  2.6   ,  0.098 , 25.    , 67.    ,0.9968,  3.2   ,  0.68  ,  9.8   ])
#result_CNN(x_train, model)

result_cnn=result_CNN(x_test,model)
cnn_conf_matrix = confusion_matrix(y_test, result_cnn)
cnn_acc_score = accuracy_score(y_test, result_cnn)
print(cnn_conf_matrix)
print(cnn_acc_score*100)


"""
Random Forest
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
df = data

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(max_depth=14,max_features=3,min_samples_leaf=3,min_samples_split=8,n_estimators=300)
rf.fit(x_train,y_train)

"""
def final_pre(input_array):#input array should be n*11
    result = []
    for item in input_array:
        res = rf.predict(item)
        if int(res) >= 1.5:
            result.append(2)
        elif int(res) >= 0.5 and int(res) <1.5:
            result.append(1)
        else:
            int(res) < 0.5
            result.append(0)
    return result
"""

def rf_predict(input):
  return [int(x) for x in rf.predict(input)]

"""Ensemble"""

svm_pred=rbf_svc.predict(x_train)
mlp_pred=NNPredict(model_mlp,x_train)
cnn_pred=result_CNN(x_train,model)
rf_pred=rf_predict(x_train)
svm_p=rbf_svc.predict(x_test)
mlp_p=NNPredict(model_mlp,x_test)
cnn_p=result_CNN(x_test,model)
rf_p=rf_predict(x_test)

ini_results=[[svm_pred[i],mlp_pred[i],cnn_pred[i],rf_pred[i]] for i in range(len(svm_pred))]
ini_r=[[svm_p[i],mlp_p[i],cnn_p[i],rf_p[i]] for i in range(len(svm_p))]

from statistics import mode
mode_pred=[mode(x) for x in ini_r]
rbf_svc_conf_matrix = confusion_matrix(y_test, mode_pred)
rbf_svc_acc_score = accuracy_score(y_test, mode_pred)
print(rbf_svc_conf_matrix)
print(rbf_svc_acc_score*100)

a=[i/np.sum(i) for i in rbf_svc_conf_matrix]
#sns.heatmap(nb_conf_matrix)
lab=["Poor","Medium","Good"]
sns.heatmap(a,xticklabels=lab,yticklabels=lab,annot=True)

stack_svc=SVC()
stack_svc.fit(ini_results,y_train)
stack_svc_pred=stack_svc.predict(ini_r)
rbf_svc_conf_matrix = confusion_matrix(y_test, stack_svc_pred)
rbf_svc_acc_score = accuracy_score(y_test, stack_svc_pred)
print(rbf_svc_conf_matrix)
print(rbf_svc_acc_score*100)

from sklearn.model_selection import GridSearchCV
param = {
    'C': [0.1,0.3,0.5,0.6,0.8,1,1.1,1.3,1.5,1.7],
    'kernel':['rbf','linear'],
    'gamma' :[0.1,0.3,0.5,0.6,0.8,0.9,1,1.1,1.2,1.3,1.4]
}
grid_svc = GridSearchCV(stack_svc, param_grid=param, scoring='accuracy', cv=10)
grid_svc.fit(ini_results, y_train)
print(grid_svc.best_params_)

stack_svc=SVC(C=0.1,gamma=0.5,kernel="rbf")
stack_svc.fit(ini_results,y_train)
stack_svc_pred=stack_svc.predict(ini_r)
rbf_svc_conf_matrix = confusion_matrix(y_test, stack_svc_pred)
rbf_svc_acc_score = accuracy_score(y_test, stack_svc_pred)
print(rbf_svc_conf_matrix)
print(rbf_svc_acc_score*100)

def Bayesian(ini_results, x_test, y_test, x_train, y_train, ini_r):
  p_y0, p_y1, p_y2 = 0, 0, 0
  y_train = y_train.tolist()
  for i in range(len(y_train)):

    if y_train[i]==0:
      p_y0+=1
    if y_train[i]==1:
      p_y1+=1
    if y_train[i]==2:
      p_y2+=1

  result_Bayesian = []
  p_y0 = p_y0/len(y_train)
  p_y1 = p_y1/len(y_train)
  p_y2 = p_y2/len(y_train)



  for i in range(len(ini_r)):
    p_x_y0 = [0,0,0,0]
    p_x_y1 = [0,0,0,0]
    p_x_y2 = [0,0,0,0]
    for j in range(len(ini_results)):
      for k in range(4):
        if ini_r[i][k] == ini_results[j][k]:
          if y_train[j] == 0:
            p_x_y0[k]+=1
          if y_train[j] == 1:
            p_x_y1[k]+=1
          if y_train[k] == 2:
            p_x_y2[k]+=1

    P_x_y0, P_x_y1, P_x_y2 = 1,1,1

    for j in range(4):
      P_x_y0 = P_x_y0*p_x_y0[j]
      P_x_y1 = P_x_y1*p_x_y1[j]
      P_x_y2 = P_x_y2*p_x_y2[j]

    p_y0_x = p_y0 * P_x_y0
    p_y1_x = p_y1 * P_x_y1
    p_y2_x = p_y2 * P_x_y2

    if max(p_y0_x, p_y1_x, p_y2_x) == 0:
      result_Bayesian.append(1)
    else:
      if max(p_y0_x, p_y1_x, p_y2_x) == p_y0_x:
        result_Bayesian.append(0)
      if max(p_y0_x, p_y1_x, p_y2_x) == p_y1_x:
        result_Bayesian.append(1)
      if max(p_y0_x, p_y1_x, p_y2_x) == p_y2_x:
        result_Bayesian.append(2)
    c=0
    for i in range(len(result_Bayesian)):
      if result_Bayesian[i] == y_train[i]:
        c+=1
  return result_Bayesian,c/len(result_Bayesian)


Bayesian(ini_results, x_test, y_test, x_train, y_train, ini_r)[-1]

import numpy as np
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import brier_score_loss as BS,recall_score,roc_auc_score as AUC

name = ["Multinomial","Gaussian","Bernoulli"]
models = [MultinomialNB(),GaussianNB(),BernoulliNB()]

for name,clf in zip(name,models):
    if name != "Gaussian":
        kbs = KBinsDiscretizer(n_bins=10, encode='onehot').fit(ini_results)
        Xtrain = kbs.transform(ini_results)
        Xtest = kbs.transform(ini_r)

    clf.fit(ini_results, y_train)
    y_pred = clf.predict(ini_r)
    proba = clf.predict_proba(ini_r)[:,1]
    score = clf.score(ini_r,y_test)
    print(y_pred)
    print(name)
    print("\tAccuracy:{:.3f}".format(score))

clf = GaussianNB()
clf.fit(ini_results, y_train)
y_pred_GaussianNB = clf.predict(ini_r)
y_pred_GaussianNB

def acc(x,y):
  count0=0
  count1=0
  count2=0
  for i in range(len(x)):
    if x[i]==0:
      if y[i]==0:
        count0=count0+1
    elif x[i]==1:
      if y[i]==1:
        count1+=1
    else:
      if y[i]==2:
        count2+=1
  print(count0/x.count(0))
  print(count1/x.count(1))
  print(count2/x.count(2))
acc(y_test, y_pred_GaussianNB)

rbf_svc_conf_matrix = confusion_matrix(y_test, y_pred_GaussianNB)
#rbf_svc_acc_score = accuracy_score(y_test, y_pred_GaussianNB)
print(rbf_svc_conf_matrix)
#print(rbf_svc_acc_score*100)
a=[i/np.sum(i) for i in rbf_svc_conf_matrix]
#sns.heatmap(nb_conf_matrix)
lab=["Poor","Medium","Good"]
sns.heatmap(a,xticklabels=lab,yticklabels=lab,annot=True)



