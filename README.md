# Ex-6-Handwritten Digit Recognition using MLP
## Aim:
       To Recognize the Handwritten Digits using Multilayer perceptron.
##  EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## Algorithm :

1. Import the necessary libraries of python.
2. In the end_to_end function, first calculate the similarity between the inputs and the peaks.
Then, to find w used the equation Aw= Y in matrix form. Each row of A (shape: (4, 2)) consists
of
3. index[0]: similarity of point with peak1 index[1]: similarity of point with peak2 index[2]: Bias
input (1) Y: Output associated with the input (shape: (4, )) W is calculated using the same
equation we use to solve linear regression using a closed solution (normal equation).
4. This part is the same as using a neural network architecture of 2-2-1, 2 node input (x1, x2)
(input layer) 2 node (each for one peak) (hidden layer) 1 node output (output layer)
5. To find the weights for the edges to the 1-output unit. Weights associated would be: edge
joining 1st node (peak1 output) to the output node edge joining 2nd node (peak2 output) to
the output node bias edge


## Program:
```python
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
data = pd.read_csv("train.csv")
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)
data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255
data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape
Y_train
def init_params():
W1 = np.random.rand(10, 784) - 0.5
b1 = np.random.rand(10, 1) - 0.5
W2 = np.random.rand(10, 10) - 0.5
b2 = np.random.rand(10, 1) - 0.5
return W1, b1, W2, b2
def ReLU(Z):
return np.maximum(Z, 0)
def softmax(Z):
A = np.exp(Z) / sum(np.exp(Z))
return A
def forward_prop(W1, b1, W2, b2, X):
Z1 = W1.dot(X) + b1
A1 = ReLU(Z1)
Z2 = W2.dot(A1) + b2
A2 = softmax(Z2)
return Z1, A1, Z2, A2
def ReLU_deriv(Z):
return Z > 0
def one_hot(Y):
one_hot_Y = np.zeros((Y.size, Y.max() + 1))
one_hot_Y[np.arange(Y.size), Y] = 1
one_hot_Y = one_hot_Y.T
return one_hot_Y
def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
one_hot_Y = one_hot(Y)
dZ2 = A2 - one_hot_Y
dW2 = 1 / m * dZ2.dot(A1.T)
db2 = 1 / m * np.sum(dZ2)
dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
dW1 = 1 / m * dZ1.dot(X.T)
db1 = 1 / m * np.sum(dZ1)
return dW1, db1, dW2, db2
def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
W1 = W1 - alpha * dW1
b1 = b1 - alpha * db1
W2 = W2 - alpha * dW2
b2 = b2 - alpha * db2
return W1, b1, W2, b2
def get_predictions(A2):
return np.argmax(A2, 0)
def get_accuracy(predictions, Y):
print(predictions, Y)
Output :
return np.sum(predictions == Y) / Y.size
def gradient_descent(X, Y, alpha, iterations):
W1, b1, W2, b2 = init_params()
for i in range(iterations):
Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
if i % 10 == 0:
print("Iteration: ", i)
predictions = get_predictions(A2)
test_prediction(0, W1, b1, W2, b2)print(get_accuracy(predictions, Y))
return W1, b1, W2, b2
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 500)
def make_predictions(X, W1, b1, W2, b2):
A2 = forward_prop(W1, b1, W2, b2, X)
predictions = get_predictions(A2)
return predictions
def test_prediction(index, W1, b1, W2, b2):
current_image = X_train[:, index, None]
prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
label = Y_train[index]
print("Prediction: ", prediction)
print("Label: ", label)
current_image = current_image.reshape((28, 28)) * 255
plt.gray()
plt.imshow(current_image, interpolation='nearest')
plt.show()
test_prediction(0, W1, b1, W2, b2)
```
## Output :
!![6(2)](https://user-images.githubusercontent.com/112338645/205050700-75923c98-95dc-4e04-8a3d-c07637af0e88.png)
[6(1![6(3)](https://user-images.githubusercontent.com/112338645/205050740-8bab0efb-fc97-4701-bb32-5ebd0a95c3b6.png)
)](![6(4)](https://user-images.githubusercontent.com/112338645/205050781-c7552fb3-f30a-4718-9e12-8d6987d472c1.png)
http![6(5)](https://user-images.githubusercontent.com/112338645/205050821-97302c08-7557-4036-a514-d05eb6f4b8de.png)
s://user![6(6)](https://user-images.githubusercontent.com/112338645/205050845-d1e815ac-de82-4911-8c6a-802f843ae805.png)
-images.githubusercontent.com/112338645/205049924-2334672e-4f39-4643-a977-083cdbd95eaf.png)

## Result:
Thus The Implementation of Handwritten Digit Recognition using MLP Is Executed Successfully
