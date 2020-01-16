import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init

import sklearn

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

epochs = 5
batch_size = 128
learning_rate = 0.001
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

x = torch.FloatTensor(X_train.tolist()).to(device)
y = torch.LongTensor(y_train.tolist()).to(device)
train = torch.utils.data.TensorDataset(x, y.reshape(-1))
train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

#y = y.squeeze_()

x_test = torch.FloatTensor(X_test.tolist()).to(device)
y_test = torch.LongTensor(y_test.tolist()).to(device)
test = torch.utils.data.TensorDataset(x_test, y_test.reshape(-1))
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True)


input_size = XData.shape[1]
output_size = 2
hidden_size = 100


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.l1 = nn.Linear(input_size, 200)
        init.xavier_normal_(self.l1.weight.data)
        init.normal_(self.l1.bias.data)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)

        self.l2 = nn.Linear(200, 200)
        init.xavier_normal_(self.l2.weight.data)
        init.normal_(self.l2.bias.data)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

        self.l3 = nn.Linear(200, 200)
        init.xavier_normal_(self.l3.weight.data)
        init.normal_(self.l3.bias.data)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)

        self.l4 = nn.Linear(200, 200)
        init.xavier_normal_(self.l4.weight.data)
        init.normal_(self.l4.bias.data)
        self.relu4 = nn.LeakyReLU(0.2, inplace=True)

        self.l5 = nn.Linear(200, output_size)
        init.xavier_normal_(self.l5.weight.data)
        init.normal_(self.l5.bias.data)
        self.relu5 = nn.LeakyReLU(0.2, inplace=True)

        self.th = nn.Tanh()

    def forward(self, input, training=True):
        input = F.dropout(input, training=training)
        input = self.l1(input)
        input = self.relu1(input)
        input = F.dropout(input, training=training)
        input = self.l2(input)
        input = self.relu2(input)
        # input = F.dropout(input, training=training)
        # input = self.l3(input)
        # input = self.relu3(input)
        # input = F.dropout(input, training=training)
        # input = self.l4(input)
        # input = self.relu4(input)
        input = F.dropout(input, training=training)
        input = self.l5(input)
        input = self.relu5(input)
        # input = self.th(input)
        return input


model = Network()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

print("Training The Model...")

# model.load_state_dict(torch.load("model", map_location='cpu'))

for e in range(epochs):
    for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
        # x_mini = x[i:i + batch_size].to(device)
        # y_mini = y[i:i + batch_size].reshape(-1).to(device)

        x_var = Variable(X_batch).to(device)
        y_var = Variable(y_batch).to(device)

        optimizer.zero_grad()
        net_out = model(x_var)

        loss = criterion(net_out, y_var)
        loss.backward()
        optimizer.step()

    print('Epoch: {} - Loss: {:.6f}'.format(e, loss.data))
    torch.save(model.state_dict(), "model")

correct = 0
total_len = 0

for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
    # x_mini = x[i:i + batch_size].to(device)
    # y_mini = y[i:i + batch_size].reshape(-1).to(device)

    x_var = Variable(X_batch).to(device)
    y_var = Variable(y_batch).to(device)
    total_len += len(y_var)

    net_out = model(x_var)
    _, idx = net_out.max(1)
    # result = y_var * 0.9 < net_out < y_var * 1.1
    correct += (idx == y_var).sum()

final_acc = float(correct * 100) / total_len
print("Train Accuracy  = {}".format(final_acc))

correct = 0
total_len = 0
recall = 0
precision = 0
temp1 = 0
for batch_idx, (X_batch, y_batch) in enumerate(test_loader):
    # x_mini = x[i:i + batch_size].to(device)
    # y_mini = y[i:i + batch_size].reshape(-1).to(device)

    x_var = Variable(X_batch).to(device)
    y_var = Variable(y_batch).to(device)
    total_len += len(y_var)

    net_out = model(x_var)
    _, idx = net_out.max(1)
    # result = y_var * 0.9 < net_out < y_var * 1.1
    correct += (idx == y_var).sum()

    recall += y_var.sum()
    precision += idx.sum()
    for index in range(len(y_var)):
        if y_var[index] == 1 and idx[index] == 1:
            temp1 += 1


final_acc = float(correct * 100) / total_len
print("Test Accuracy  = {}".format(final_acc))
print("Precision = {}".format(temp1 * 100 / precision))
print("Recall = {}".format(temp1 * 100 / recall))