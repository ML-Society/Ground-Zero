import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
sns.set_style('white')
#%matplotlib inline
#hello

plt.grid()

import pandas as pd

df = pd.read_csv("bimodal_example.csv")
mean = df.mean(axis=0)
print('mean', mean)

#show first example
data = df.x

#plot histogram
g = sns.distplot(data, bins=20, kde=False)
plt.show()

# E step - given current params, estimate a pd
# M step - given current data, estimate parameters to update
class Gaussian:
    'Model Gaussian'
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        self.dims = len(mu)

    def pdf(self, datum):   # datum is the value of the variable observed
        'Probability of a point given the current parameters'
        u = (datum - self.mu)/abs(self.sigma)
        y = (1 / (sqrt(2*pi))*abs(self.sigma))
        return y

    def __repr__(self):
        return 'Gaussian({0:4.6}, {1:4.6})'.format(self.mu, self.sigma)
                        # ^ zeroth formatter,
                        # padding indeted by 4 whitespaces from left,
                        # truncated after the 6th value

def visualise(model_params):
    fig = plt.figure()
    ax = fig.add_subplot(111)



best_single = Gaussian(np.mean(data), np.std(data))
#print('Best single Gaussian: mu = {:.2}, sigma = {:.2}'.format(best_single.mu, best_single.sigma))

# fit a single gaussian to the data
