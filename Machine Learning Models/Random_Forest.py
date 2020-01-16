import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init
import torch.utils.data
import sklearn
from sklearn import preprocessing
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestRegressor
import math 

df = pd.read_csv("SCADA.CSV")
data = df.drop(['Address','CommandResponse','ControlMode','ControlScheme','InvalidDataLength','InvalidFunctionCode','deltaPIDCycleTime','deltaPIDDeadband','deltaPIDGain','deltaPIDRate','deltaPIDReset','deltaSetPoint','PIDCycleTime','PIDDeadband','PIDGain','PIDRate','PIDReset','SetPoint'], axis=1)
label_encoder = LabelEncoder()
data['FunctionCode'] = label_encoder.fit_transform(data['FunctionCode'])
data = data[data.PumpState != 'X']
data = data[data.SolenoidState != 'X']
data = data.replace({'output' : { 'Good' : 1, 'Burst' : 0, 'Fast' : 0, 'Negative' : 0, 'Setpoint' : 0,'Single' : 0, 'Slow' : 0, 'Wave' : 0}})
y = data.output
x= data.drop(['output'], axis=1)
x['PumpState'] = label_encoder.fit_transform(x['PumpState'])
x['SolenoidState'] = label_encoder.fit_transform(x['SolenoidState'])

XData = np.array(x)
YData = np.array(y)

scaling = MinMaxScaler(feature_range=(-1,1)).fit(XData)
XData = scaling.transform(XData)

scaler = sklearn.preprocessing.StandardScaler().fit(XData)
XData = scaler.transform(XData)

X_train, X_test, y_train, y_test = train_test_split(XData, YData, test_size=0.2, shuffle=True, random_state=1)
model = RandomForestRegressor(n_estimators=100, n_jobs=-1)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))


y_pred_MLP = model.predict(X_test)

def plot_confusion_matrix(y_true,y_pred):
    cm_array = confusion_matrix(y_true,y_pred)
    confusion = cm_array
    print("----- confusion matrix ----------")
    print(confusion)
    print('--------- Parameters-------------------------')
    Precision = confusion[1, 1] / sum(confusion[ :,1])
    print('Precision  is %0.4f' % Precision)
    Recall = confusion[1, 1] / sum(confusion[1,:])
    print('recall is %0.4f' % Recall)
    F_score = 2 * Precision * Recall / (Precision + Recall)
    print('F_score  is %0.4f' % F_score)
    Accuracy = (confusion[1, 1] + confusion[0, 0]) / (confusion[1, 1] + confusion[0, 0]+ confusion[1, 0] + confusion[0, 1])
    print('Accuracy  is %0.4f'% Accuracy)
    MCC = ((confusion[0, 0] * confusion[1, 1])-(confusion[0, 1])*(confusion[1, 0])) / (math.sqrt((confusion[0, 0]+confusion[1, 0])*(confusion[0, 0]+confusion[0, 1]) *(confusion[1, 1]+confusion[1, 0]) *(confusion[1, 1]+confusion[0, 1])))
    print ('MCC  is %0.4f'% MCC)

    true_labels = np.unique(y_true)
    pred_labels = np.unique(y_pred)
    plt.imshow(cm_array, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix", fontsize=16)
    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    cbar.set_label('Count ', rotation=270, labelpad=30, fontsize=12)
    xtick_marks = np.arange(len(true_labels))
    ytick_marks = np.arange(len(pred_labels))
    plt.xticks(xtick_marks, true_labels, rotation=90)
    plt.yticks(ytick_marks,pred_labels)
    plt.tight_layout()
    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predicted label', fontsize=14)
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 12
    fig_size[1] = 12
    plt.rcParams["figure.figsize"] = fig_size
    plt.show()
plot_confusion_matrix(y_test,y_pred_MLP.round())

