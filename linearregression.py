import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable


def makedata(numdatapoints):
    x = np.linspace(-10, 10, numdatapoints)

    coeffs = [0.5, 5]

    y = np.polyval(coeffs, x) + 1 * np.random.randn(numdatapoints)
    '''
    inputs = np.zeros((2, numdatapoints))
    inputs[0] = x
    inputs[1] = y
    '''
    return x, y

def makefeatures(powers):
    features = np.ones((inputs.shape[0], len(powers)))
    for i in range(len(powers)):
        features[:, i] = (inputs**powers[i])
    print(features.shape)

    return features.T

def scale_features(features):
    scaled = features

    avg = np.mean(scaled[:], axis=1).reshape(-1, 1)
    ranges = np.ptp(scaled[:], axis=1).reshape(-1, 1)

    scaled[:] -= avg
    scaled[:] = np.divide(scaled[:], ranges)

    return scaled, avg, ranges


numdatapoints = 10
inputs, labels = makedata(numdatapoints)

# create our plotting objects
fig = plt.figure(figsize=(10, 20))

ax1 = fig.add_subplot(121)
ax1.set_xlabel('Input')
ax1.set_ylabel('Output')

ax1.scatter(np.array(inputs), np.array(labels))
ax1.grid()

plt.ion()
plt.show()

ax2 = fig.add_subplot(122)
ax2.set_title('Error vs Epochs')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Error')
ax2.grid()

# initialise the prediction line with correctly dimensioned data
line1, = ax1.plot(inputs, inputs)#, inputs[0], 'g')
print(inputs)


class LinearModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.l = torch.nn.Linear(features.shape[0], 1)

    def forward(self, x):
        out = self.l(x)
        return out


# hyperparameters
epochs = 100
lr = 0.1
batch_size = 3

powers = [1, 2, 3]

features = makefeatures(powers)
scaled, avg, ranges = scale_features(features)

datain = Variable(torch.Tensor(features.T))
labels = Variable(torch.Tensor(labels.T))
mymodel = LinearModel()

criterion = torch.nn.MSELoss(size_average=True)
optimiser = torch.optim.SGD(mymodel.parameters(), lr=lr)

def train():
    costs=[]
    for e in range(epochs):

        prediction = mymodel(datain)
        print(prediction.shape)



        cost = criterion(prediction, labels)

            costs.append(cost.data)

        print('Epoch:', e, 'Cost:', cost.data[0])


        params = [mymodel.state_dict()[i][0] for i in mymodel.state_dict()]

        weights = params[0]
        bias = params[1]

        optimiser.zero_grad()
        cost.backward()
        optimiser.step()

        line1.set_ydata(torch.mm(weights.view(1, -1), datain.data.t()) + bias)
        fig.canvas.draw()
        ax2.plot(costs)

train()

print(mymodel.state_dict())
