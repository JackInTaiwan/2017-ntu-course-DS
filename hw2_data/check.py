import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


### Load data and transfer them into numpy
data = pd.read_csv('spambase.csv', header=1, index_col=0)
x = np.array(data.iloc[:, :-1])
y = np.array(data.iloc[:, -1])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.16)


data = np.array(pd.read_csv('example_test.csv', header=None))
print (data)
print (data.shape)